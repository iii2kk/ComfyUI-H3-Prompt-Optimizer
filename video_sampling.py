import base64
from io import BytesIO
import math

import av
from PIL import Image


def sampling_times(start_sec, end_sec, fps, max_frames):
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    count = max(1, int(math.floor((end_sec - start_sec) * float(fps))) + 1)
    count = min(count, int(max_frames))
    if count == 1:
        return [start_sec]
    return [start_sec + (end_sec - start_sec) * index / (count - 1) for index in range(count)]


def _encode_frame(frame, max_edge=768):
    image = frame.to_image().convert("RGB")
    longest = max(image.size)
    if longest > max_edge:
        scale = max_edge / longest
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=86)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sample_video(path, start_sec, end_sec, fps, max_frames=48, cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Video sampling was cancelled.")
    targets = sampling_times(start_sec, end_sec, fps, max_frames)
    selected = []
    target_index = 0
    last_frame = None
    last_time = 0.0

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError("Generation artifact has no video stream.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate or 24.0)
        for frame_index, frame in enumerate(container.decode(stream)):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Video sampling was cancelled.")
            timestamp = float(frame.time) if frame.time is not None else frame_index / rate
            last_frame = frame
            last_time = timestamp
            while target_index < len(targets) and timestamp >= targets[target_index]:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Video sampling was cancelled.")
                selected.append({
                    "timestamp": targets[target_index],
                    "jpeg_base64": _encode_frame(frame),
                })
                target_index += 1
            if target_index >= len(targets):
                break

    if last_frame is None:
        raise ValueError("Generation artifact has no decodable video frames.")
    while target_index < len(targets):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Video sampling was cancelled.")
        selected.append({
            "timestamp": min(targets[target_index], last_time),
            "jpeg_base64": _encode_frame(last_frame),
        })
        target_index += 1
    return selected
