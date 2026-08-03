import gzip

from flask import Flask, Response

from atcroster.compression import register_response_compression


def _app():
    application = Flask(__name__)
    register_response_compression(application, minimum_size=100)

    @application.get("/large")
    def large():
        return "roster-data" * 500

    @application.get("/events")
    def events():
        return Response(iter(("data: ready\n\n",)), mimetype="text/event-stream")

    return application


def test_large_html_response_is_gzipped_when_supported():
    response = _app().test_client().get(
        "/large", headers={"Accept-Encoding": "gzip"}
    )
    assert response.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(response.data).startswith(b"roster-data")
    assert "Accept-Encoding" in response.headers["Vary"]


def test_streaming_event_response_is_never_compressed():
    response = _app().test_client().get(
        "/events", headers={"Accept-Encoding": "gzip"}
    )
    assert "Content-Encoding" not in response.headers
