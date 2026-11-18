# Local research interface

This Flask interface adapts CuraView's existing detection, evaluation, graph query, and log views to the public runtime.

```sh
python -m pip install -r webui/requirements.txt
python -m webui.app
```

Open `http://127.0.0.1:5000`. Set up the [detector](../detection/) and a GraphRAG 2.7 index before submitting detection or graph requests. The evaluation page computes sentence micro and patient macro metrics over caller-supplied paired labels and predictions. Its two JSON arrays use the matching columns `patient_id` and `sentence_id`. The reference objects provide `y_true` and predictions provide `y_pred` as binary labels.

The app runs one queued detection at a time. Task status uses the detector's completed, partial, or failed state. Task IDs stay available until restart. Reports and logs live in `runs/webui/<task ID>/` by default. Set `CURAVIEW_WEBUI_RUN_ROOT` to use a different private directory. Graph queries also retain traces in that directory. The UI serves its browser assets locally and binds only to loopback.

```sh
python -m webui.smoke_test
```

The smoke test uses an in-process Flask client and replaces the detector and GraphRAG subprocess boundaries for successful and failed cases. Real provider-backed requests use model endpoints in your own detector and graph settings.
