"""
Adaptive Neuro-AI - Streamlit dashboard.

Run with:  streamlit run src/app.py

The UI holds no algorithmic logic. It renders state produced by
core.bci_engine and core.self_healing, which keeps the science testable and
the interface replaceable.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.bci_engine import SIMULATED_DEVICES, BCIEngine       # noqa: E402
from src.core.config import CLASS_NAMES, MODEL_PATH, ensure_directories  # noqa: E402
from src.core.mouse_control import MouseController                 # noqa: E402

CLASS_COLOURS = {"LEFT": "#2563eb", "RIGHT": "#16a34a", "REST": "#f59e0b"}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def init_state() -> None:
    ensure_directories()
    defaults = {
        "engine": None,
        "mouse": None,
        "mouse_enabled": False,
        "inject_fault": False,
        "self_healing": True,
        "edf_results": None,
        "session_running": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state.engine is None:
        st.session_state.engine = BCIEngine(MODEL_PATH, self_healing=True)
    if st.session_state.mouse is None:
        st.session_state.mouse = MouseController()


# --------------------------------------------------------------------------
# Shared widgets
# --------------------------------------------------------------------------
def probability_chart(probabilities: np.ndarray) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=CLASS_NAMES,
            y=[float(p) for p in probabilities],
            marker_color=[CLASS_COLOURS[name] for name in CLASS_NAMES],
            text=[f"{p:.2f}" for p in probabilities],
            textposition="auto",
        )
    )
    figure.update_layout(
        yaxis_range=[0, 1], height=300, showlegend=False,
        margin=dict(t=40, b=30), title="Class probabilities",
    )
    return figure


def signal_chart(window: np.ndarray, max_channels: int = 4) -> go.Figure:
    figure = go.Figure()
    n = min(max_channels, window.shape[0])
    t = np.linspace(0, 4, window.shape[1])
    for index in range(n):
        figure.add_trace(
            go.Scatter(x=t, y=window[index] + index * 6, mode="lines",
                       name=f"Ch {index + 1}", line=dict(width=1.4))
        )
    figure.update_layout(
        height=300, title="Incoming EEG window (offset for readability)",
        xaxis_title="Time (s)", yaxis_title="Amplitude (a.u.)",
        margin=dict(t=40, b=30),
    )
    return figure


def healing_panel(engine: BCIEngine) -> None:
    status = engine.healer.status()
    st.subheader("Self-healing status")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean confidence", f"{status['mean_confidence']:.2f}")
    col2.metric("Mean entropy", f"{status['mean_entropy']:.2f}")
    col3.metric("Channels repaired", status["channels_repaired"])
    col4.metric("Model adaptations", status["adaptations"])

    if not status["enabled"]:
        st.warning("Self-healing is disabled. The decoder will not recover from drift.")
    elif status["drift_detected"]:
        st.error("Drift detected - the adaptation routine will fine-tune the decoder.")
    else:
        st.success("Signal quality and decoder confidence are within tolerance.")

    if engine.healer.events:
        with st.expander(f"Healing log ({len(engine.healer.events)} events)"):
            for event in reversed(engine.healer.events[-15:]):
                stamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
                st.write(f"`{stamp}` **{event.kind}** - {event.detail}")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_realtime() -> None:
    engine: BCIEngine = st.session_state.engine
    st.header("Real-time BCI interface")

    controls, output = st.columns([1, 2])

    with controls:
        st.subheader("Session controls")
        intent = st.selectbox(
            "Simulated user intent",
            ["Random"] + CLASS_NAMES,
            help="Which motor imagery the simulated subject is performing.",
        )
        engine.noise_level = st.slider("Noise level", 0.0, 2.0, engine.noise_level, 0.05)
        st.session_state.inject_fault = st.checkbox(
            "Inject electrode fault",
            value=st.session_state.inject_fault,
            help="Kills two electrodes and saturates a third, so the self-healing "
                 "layer can be observed repairing them.",
        )
        healing = st.checkbox("Enable self-healing", value=st.session_state.self_healing)
        st.session_state.self_healing = healing
        engine.healer.enabled = healing

        st.markdown("**Cursor control**")
        if st.button("Enable cursor control"):
            ok, message = st.session_state.mouse.initialise()
            st.session_state.mouse_enabled = ok
            (st.success if ok else st.warning)(message)
        if st.session_state.mouse_enabled:
            st.caption("Cursor control is active. Move the pointer to a screen "
                       "corner to trigger the pyautogui fail-safe.")

        steps = st.number_input("Windows to decode", 1, 200, 10)
        run = st.button("Run decoding session", type="primary")
        if st.button("Reset engine"):
            engine.reset()
            st.info("Prediction history cleared and weights restored to baseline.")

    with output:
        placeholder_probability = st.empty()
        placeholder_signal = st.empty()
        placeholder_label = st.empty()

        if run:
            progress = st.progress(0.0)
            for step in range(int(steps)):
                requested = None if intent == "Random" else intent
                if requested is None:
                    requested = CLASS_NAMES[np.random.randint(len(CLASS_NAMES))]
                window = engine.simulate_window(requested, st.session_state.inject_fault)
                prediction = engine.predict_window(window, source="SIMULATED")

                if st.session_state.mouse_enabled:
                    st.session_state.mouse.move(prediction.class_index)

                placeholder_label.markdown(
                    f"### Decoded command: `{prediction.label}` "
                    f"({prediction.confidence:.2f} confidence)"
                )
                placeholder_probability.plotly_chart(
                    probability_chart(prediction.probabilities),
                    use_container_width=True, key=f"prob_{step}",
                )
                placeholder_signal.plotly_chart(
                    signal_chart(window), use_container_width=True, key=f"sig_{step}"
                )
                progress.progress((step + 1) / int(steps))
                time.sleep(0.15)
            progress.empty()
        else:
            placeholder_label.markdown(
                f"### Last decoded command: `{engine.current_prediction}`"
            )
            placeholder_probability.plotly_chart(
                probability_chart(engine.probabilities), use_container_width=True
            )

    st.divider()
    healing_panel(engine)

    st.divider()
    st.subheader("Prediction history")
    if engine.history:
        recent = engine.history[-40:]
        base = recent[0].timestamp
        figure = go.Figure()
        for name in CLASS_NAMES:
            points = [(p.timestamp - base, p.confidence) for p in recent if p.label == name]
            if points:
                figure.add_trace(
                    go.Scatter(
                        x=[p[0] for p in points], y=[p[1] for p in points],
                        mode="markers", name=name,
                        marker=dict(size=11, color=CLASS_COLOURS[name]),
                    )
                )
        figure.update_layout(height=280, xaxis_title="Time (s)",
                             yaxis_title="Confidence", yaxis_range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("No predictions yet. Run a decoding session to populate the timeline.")


def page_edf() -> None:
    engine: BCIEngine = st.session_state.engine
    st.header("EDF file analysis")
    st.info(
        "Upload a PhysioNet `.edf` recording. Every annotated 4-second epoch is "
        "band-pass filtered, normalised and classified as LEFT, RIGHT or REST."
    )

    uploaded = st.file_uploader("EDF recording", type=["edf"])
    if uploaded is not None and st.button("Analyse recording", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as handle:
            handle.write(uploaded.getvalue())
            temporary_path = handle.name
        try:
            with st.spinner("Processing recording..."):
                results = engine.analyse_edf(temporary_path)
            results["filename"] = uploaded.name
            st.session_state.edf_results = results
            st.success(f"Analysed {results['n_epochs']} epochs from {uploaded.name}.")
        except Exception as exc:                          # noqa: BLE001
            st.error(f"Could not process this recording: {exc}")
        finally:
            os.unlink(temporary_path)

    results = st.session_state.edf_results
    if not results:
        return

    predictions = results["predictions"]
    confidences = results["confidences"]
    total = len(predictions)
    counts = {name: predictions.count(name) for name in CLASS_NAMES}

    st.subheader(f"Results - {results['filename']}")
    columns = st.columns(4)
    columns[0].metric("Epochs", total)
    for column, name in zip(columns[1:], CLASS_NAMES):
        column.metric(name, counts[name], f"{counts[name] / total * 100:.1f}%")

    if results["true_labels"].size == total:
        accuracy = float(
            np.mean(np.array([CLASS_NAMES.index(p) for p in predictions]) == results["true_labels"])
        )
        st.metric("Accuracy against file annotations", f"{accuracy * 100:.2f}%")

    left, right = st.columns(2)
    with left:
        figure = go.Figure(go.Histogram(x=confidences, nbinsx=20, marker_color="#2563eb"))
        figure.update_layout(title="Confidence distribution", height=300,
                             xaxis_title="Confidence", yaxis_title="Epochs")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = go.Figure(
            go.Pie(labels=CLASS_NAMES, values=[counts[n] for n in CLASS_NAMES],
                   marker_colors=[CLASS_COLOURS[n] for n in CLASS_NAMES])
        )
        figure.update_layout(title="Predicted class distribution", height=300)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Per-epoch detail")
    table = [
        {
            "Epoch": index + 1,
            "Prediction": predictions[index],
            "Confidence": f"{confidences[index]:.3f}",
            "LEFT": f"{results['probabilities'][index][0]:.3f}",
            "RIGHT": f"{results['probabilities'][index][1]:.3f}",
            "REST": f"{results['probabilities'][index][2]:.3f}",
            "Healed": "yes" if results["healed_flags"][index] else "",
        }
        for index in range(min(50, total))
    ]
    st.dataframe(table, use_container_width=True)
    healing_panel(engine)


def page_devices() -> None:
    engine: BCIEngine = st.session_state.engine
    st.header("Device settings")

    discovered = engine.scan_devices()
    if discovered:
        st.success(f"Hardware found: {', '.join(discovered)}")
    else:
        st.info(
            "No physical EEG hardware is attached. The system runs on recorded "
            "PhysioNet data and on the built-in signal simulator. Hardware "
            "support plugs into `BCIEngine.scan_devices`."
        )

    choice = st.selectbox("Simulated device profile", list(SIMULATED_DEVICES))
    columns = st.columns(2)
    if columns[0].button("Connect simulated device"):
        ok, message = engine.connect_simulated_device(choice)
        (st.success if ok else st.error)(message)
    if columns[1].button("Disconnect"):
        engine.disconnect_device()
        st.info("Device disconnected.")

    info = engine.device_info()
    st.subheader("Current device")
    metrics = st.columns(4)
    metrics[0].metric("Status", "Connected" if info["connected"] else "None")
    metrics[1].metric("Channels", info["channels"])
    metrics[2].metric("Sampling rate", f"{info['sampling_rate']} Hz")
    metrics[3].metric("Battery", f"{info['battery']}%")
    st.caption(
        "Devices with fewer than 64 electrodes are tiled up to the model's "
        "64-channel input by `core.preprocessing.conform_channels`."
    )


def page_model() -> None:
    engine: BCIEngine = st.session_state.engine
    st.header("Model information")

    (st.success if engine.model_loaded else st.warning)(engine.load_message)

    st.subheader("Architecture - EEGNet-TCN")
    st.code(
        "Input  (batch, 1, 64, 640)\n"
        "Conv2d(1, 16, kernel=(1, 32))      -> BatchNorm -> ELU\n"
        "DepthwiseConv2d(16, 32, (64, 1))   -> BatchNorm -> ELU -> AvgPool(1,4) -> Dropout\n"
        "Conv2d(32, 32, kernel=(1, 16))     -> BatchNorm -> ELU -> AvgPool(1,8) -> Dropout\n"
        "Flatten -> Linear(64) -> ELU -> Linear(3)\n"
        "Softmax over [LEFT, RIGHT, REST]",
        language="text",
    )

    from src.core.model import count_parameters
    columns = st.columns(3)
    columns[0].metric("Trainable parameters", f"{count_parameters(engine.model):,}")
    columns[1].metric("Compute device", str(engine.device))
    columns[2].metric(
        "Checkpoint accuracy",
        f"{engine.checkpoint_accuracy:.2f}%"
        if isinstance(engine.checkpoint_accuracy, (int, float)) else "n/a",
    )

    st.subheader("Signal pipeline")
    st.markdown(
        "1. Band-pass filter 7-30 Hz (mu and beta rhythms)\n"
        "2. Epoch 0-4 s after each annotated event\n"
        "3. Resample to 160 Hz, fixed 640 samples\n"
        "4. Per-channel z-score, zero-variance safe\n"
        "5. Self-healing channel repair\n"
        "6. EEGNet-TCN classification"
    )


def main() -> None:
    st.set_page_config(page_title="Adaptive Neuro-AI BCI", page_icon="brain", layout="wide")
    init_state()

    st.title("Adaptive Neuro-AI")
    st.caption("A self-healing brain-computer interface for motor-imagery decoding")

    page = st.sidebar.radio(
        "Navigation",
        ["Real-time BCI", "EDF file analysis", "Device settings", "Model information"],
    )
    st.sidebar.divider()
    engine: BCIEngine = st.session_state.engine
    st.sidebar.metric("Last command", engine.current_prediction)
    st.sidebar.metric("Adaptations", engine.healer.adaptation_count)
    st.sidebar.caption(engine.load_message)

    if page == "Real-time BCI":
        page_realtime()
    elif page == "EDF file analysis":
        page_edf()
    elif page == "Device settings":
        page_devices()
    else:
        page_model()


if __name__ == "__main__":
    main()
