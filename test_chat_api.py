import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def http_get(path: str):
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_health():
    status, body = http_get("/health")
    assert status == 200, f"/health status != 200, got {status}"
    assert body.get("status") == "ok", f"/health body error: {body}"
    print("✅ /health passed")


def test_chat():
    payload = {
        "message": "你好，请用一句话介绍你自己",
        "history": [
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": "我们开始测试。"},
            {"role": "assistant", "content": "好的，请继续。"},
        ],
    }
    status, body = http_post("/chat", payload)
    assert status == 200, f"/chat status != 200, got {status}, body={body}"
    assert isinstance(body.get("reply"), str), f"reply 不是字符串: {body}"
    assert body.get("reply", "").strip(), f"reply 为空: {body}"
    print("✅ /chat passed")
    print("助手回复:", body["reply"])


def test_chat_bad_request():
    payload = {"message": "   ", "history": []}
    try:
        http_post("/chat", payload)
        raise AssertionError("预期 /chat 返回 400，但实际成功")
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        assert e.code == 400, f"预期 400，实际 {e.code}, body={body}"
        print("✅ /chat bad request passed")


def main():
    print(f"Testing API: {BASE_URL}")
    try:
        test_health()
        test_chat()
        test_chat_bad_request()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

    print("\n🎉 所有测试通过")


if __name__ == "__main__":
    main()
