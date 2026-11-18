"""Scoped HTTPX client boundary capture; no vendor files or auth headers saved.

Use in a dedicated workflow process. Every actual HTTP attempt is separate,
including SDK retries. Stream bytes are captured as the caller consumes them.
No claim is made about server-private prompts, token IDs or reasoning.
"""
import contextvars
import hashlib
import json
import threading
from contextlib import contextmanager
from urllib.parse import urlsplit, parse_qsl

import httpx
from curaview._99_tracing.storage.workflow_trace import TraceCaptureError

_store = contextvars.ContextVar('curaview_http_trace_store', default=None)
_quality = contextvars.ContextVar('curaview_http_response_quality', default=None)
_lease = threading.RLock()
_depth = 0
_owner = None
_owner_group = None
_original_sync = None
_original_async = None
SAFE_HEADERS = {'content-type', 'content-encoding', 'content-length',
                'x-request-id', 'request-id', 'retry-after', 'openai-processing-ms'}


def _headers(headers):
    return {k:v for k,v in headers.items() if k.lower() in SAFE_HEADERS}


def _metadata(request, stream):
    parsed=urlsplit(str(request.url))
    return {'method':request.method,
            'url':f'{parsed.scheme}://{parsed.hostname}' +
                  (f':{parsed.port}' if parsed.port else '') + parsed.path,
            'query':{k: v if k.lower() == 'api-version' else '[redacted]'
                     for k, v in parse_qsl(parsed.query, keep_blank_values=True)},
            'headers':_headers(request.headers), 'stream':stream,
            'boundary':'httpx.Client.send', 'server_internal_input':'unobservable',
            'auth_headers':'excluded', 'body_redaction':'authentication_only'}


def _reject_unusable(store, request, reason, response=None):
    # Latch before recording: failed persistence must never permit a next send.
    _quality.get()['failed'] = True
    with store.call('workflow.response_quality_gate', {
            'endpoint_path': urlsplit(str(request.url)).path,
            'reason': reason,
            'response_sha256': hashlib.sha256(response.content).hexdigest()
                if response is not None else None,
            'next_transport_dispatch_blocked': True}):
        raise RuntimeError('unusable chat response; workflow aborted: ' + reason)


def _quality_before(store, request, stream):
    scope = _quality.get()
    if scope is None:
        return
    if scope['failed']:
        raise RuntimeError('response quality scope aborted; no further dispatch')
    if stream and urlsplit(str(request.url)).path.endswith('/chat/completions'):
        _reject_unusable(store, request, 'text response gate requires nonstream chat')


def _quality_after(store, request, response):
    if (_quality.get() is None or response.status_code >= 400
            or not urlsplit(str(request.url)).path.endswith('/chat/completions')):
        return
    try:
        body = json.loads(response.content)
        choices = body.get('choices') if isinstance(body, dict) else None
        valid = (isinstance(choices, list) and len(choices) == 1
                 and isinstance(choices[0], dict)
                 and choices[0].get('finish_reason') == 'stop'
                 and isinstance(choices[0].get('message'), dict)
                 and isinstance(choices[0]['message'].get('content'), str)
                 and bool(choices[0]['message']['content'].strip()))
    except (ValueError, UnicodeError):
        valid = False
    if not valid:
        _reject_unusable(store, request,
                         'expected one nonempty text choice with finish_reason=stop', response)


class _SyncStream(httpx.SyncByteStream):
    def __init__(self, inner, store, call_id):
        self.inner,self.store,self.call_id=inner,store,call_id
        self.terminal=False

    def __iter__(self):
        try:
            for chunk in self.inner:
                self.store.record(self.call_id,'stream_chunk',chunk)
                yield chunk
            self.store.finish(self.call_id,{'stream_complete':True})
            self.terminal=True
        except GeneratorExit:
            raise  # close() registers caller interruption.
        except BaseException as error:
            if isinstance(error,TraceCaptureError):
                self.store.invalidate();self.terminal=True
                raise
            self.store.fail(self.call_id,error)
            self.terminal=True
            raise

    def close(self):
        try:
            self.inner.close()
        finally:
            if not self.terminal:
                self.store.fail(self.call_id,RuntimeError('caller closed stream before completion'))
                self.terminal=True


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, inner, store, call_id):
        self.inner,self.store,self.call_id=inner,store,call_id
        self.terminal=False

    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                self.store.record(self.call_id,'stream_chunk',chunk)
                yield chunk
            self.store.finish(self.call_id,{'stream_complete':True})
            self.terminal=True
        except GeneratorExit:
            raise
        except BaseException as error:
            if isinstance(error,TraceCaptureError):
                self.store.invalidate();self.terminal=True
                raise
            self.store.fail(self.call_id,error)
            self.terminal=True
            raise

    async def aclose(self):
        try:
            await self.inner.aclose()
        finally:
            if not self.terminal:
                self.store.fail(self.call_id,RuntimeError('caller closed stream before completion'))
                self.terminal=True


def _sync_send(client, request, *, stream=False, **kwargs):
    store=_store.get()
    if store is None:
        return _original_sync(client,request,stream=stream,**kwargs)
    _quality_before(store,request,stream)
    call_id=store.begin('api.http',_metadata(request,stream))
    response=None
    try:
        store.record(call_id,'request_body',request.read())
        # Force transport streaming internally so even non-streaming reads and
        # disconnects are observable. Reproduce HTTPX's ordinary read/close.
        response=_original_sync(client,request,stream=True,**kwargs)
        store.record(call_id,'response_headers',{'status_code':response.status_code,
                                                 'headers':_headers(response.headers)})
        if response.is_stream_consumed:
            store.record(call_id,'response_body',response.content)
            store.finish(call_id,{'stream_complete':True,'status_code':response.status_code})
        else:
            response.stream=_SyncStream(response.stream,store,call_id)
            if not stream:
                try:
                    response.read()
                finally:
                    response.close()
    except BaseException as error:
        # A wrapped stream has already registered its failure. A second failure
        # event is diagnostic, never a second terminal or fabricated success.
        if isinstance(error,TraceCaptureError):
            store.invalidate()
            raise
        if response is None or not getattr(response.stream,'terminal',False):
            store.fail(call_id,error)
        raise
    # HTTP success and model-output usability have distinct trace terminals.
    # Run after capture's exception handler to avoid failing a completed HTTP call twice.
    _quality_after(store,request,response)
    return response


async def _async_send(client, request, *, stream=False, **kwargs):
    store=_store.get()
    if store is None:
        return await _original_async(client,request,stream=stream,**kwargs)
    _quality_before(store,request,stream)
    call_id=store.begin('api.http',_metadata(request,stream))
    response=None
    try:
        store.record(call_id,'request_body',await request.aread())
        response=await _original_async(client,request,stream=True,**kwargs)
        store.record(call_id,'response_headers',{'status_code':response.status_code,
                                                 'headers':_headers(response.headers)})
        if response.is_stream_consumed:
            store.record(call_id,'response_body',response.content)
            store.finish(call_id,{'stream_complete':True,'status_code':response.status_code})
        else:
            response.stream=_AsyncStream(response.stream,store,call_id)
            if not stream:
                try:
                    await response.aread()
                finally:
                    await response.aclose()
    except BaseException as error:
        if isinstance(error,TraceCaptureError):
            store.invalidate()
            raise
        if response is None or not getattr(response.stream,'terminal',False):
            store.fail(call_id,error)
        raise
    _quality_after(store,request,response)
    return response


@contextmanager
def capture_http(store, *, stop_on_unusable_chat=False, concurrency_group=None):
    """Restore the original methods on every exit; no cross-thread inheritance.

    Async tasks inherit contextvars. Applications creating new threads must use
    copy_context().run or open this scope in the worker. Subprocesses need their
    own capture entrypoint. Unsupported clients are NOT silently called covered.

    Distinct stores may overlap only with the exact same explicit, non-None
    concurrency_group object. Each worker's ContextVar retains its actual store;
    the group is admission authority, never a replacement trace destination.
    Nested foreign stores in one context remain prohibited.

    Opt-in text-only, nonstream chat workflows can reject unusable successful
    responses after retaining their bytes, then block every later transport send
    in this scope. Nested scopes inherit the latch. Already in-flight concurrent
    requests cannot be cancelled here; bounded serial workflows avoid that gap.
    The default preserves historical capture-only behavior.
    """
    global _depth,_owner,_owner_group,_original_sync,_original_async
    with _lease:
        if _depth:
            same_admission = ((_owner_group is None and concurrency_group is None and _owner is store)
                              or (_owner_group is not None and _owner_group is concurrency_group))
            nested_store = _store.get()
            if not same_admission or (nested_store is not None and nested_store is not store):
                raise RuntimeError('overlapping HTTP capture stores are not supported without one explicit concurrency group')
        if not _depth:
            _original_sync,httpx.Client.send=httpx.Client.send,_sync_send
            _original_async,httpx.AsyncClient.send=httpx.AsyncClient.send,_async_send
            _owner=store
            _owner_group=concurrency_group
        _depth+=1
    token=_store.set(store)
    quality_token = (_quality.set({'failed': False})
                     if stop_on_unusable_chat and _quality.get() is None else None)
    try:
        yield store
    finally:
        _store.reset(token)
        if quality_token is not None:
            _quality.reset(quality_token)
        with _lease:
            _depth-=1
            if not _depth:
                httpx.Client.send=_original_sync
                httpx.AsyncClient.send=_original_async
                _owner=None
                _owner_group=None
