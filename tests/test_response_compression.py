from flask import Flask

from atcroster.compression import register_response_compression


def test_large_html_response_is_gzipped_for_supporting_browser():
    app = Flask(__name__)
    register_response_compression(app, minimum_size=100)

    @app.get("/")
    def large_page():
        return "<p>Roster content</p>" * 200

    response = app.test_client().get("/", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert "Accept-Encoding" in response.headers.getlist("Vary")
    assert int(response.headers["Content-Length"]) < 4200


def test_compression_is_not_forced_on_unsupported_browser():
    app = Flask(__name__)
    register_response_compression(app, minimum_size=100)

    @app.get("/")
    def large_page():
        return "Roster content" * 200

    response = app.test_client().get("/")

    assert "Content-Encoding" not in response.headers


def test_streaming_response_is_never_buffered_for_compression():
    app = Flask(__name__)
    register_response_compression(app, minimum_size=1)

    @app.get("/events")
    def events():
        return (item for item in ["data: ready\n\n"]), {
            "Content-Type": "text/event-stream"
        }

    response = app.test_client().get("/events", headers={"Accept-Encoding": "gzip"})

    assert response.data == b"data: ready\n\n"
    assert "Content-Encoding" not in response.headers
