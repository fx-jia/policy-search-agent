"""Real Quart requests with isolated identity/database integration boundaries."""

import functools
import importlib.util
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from quart import Blueprint, Quart, jsonify, request
from werkzeug.local import LocalProxy

from test.unit_test.policy.test_medical_assistant_service import load_module, PolicyFixture, ROOT


class MedicalAssistantApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        service = load_module()
        self.policy = PolicyFixture(self.tmp.name)
        self.host = service.MedicalAssistantService(self.policy, owner_user_id="owner", access_check=lambda u, k: True)
        self.addCleanup(self.host._owner_file.close)
        self.app = Quart(__name__)
        self.blueprint = Blueprint("medical_test", __name__)
        apps = ModuleType("api.apps")
        apps.current_user = LocalProxy(lambda: SimpleNamespace(id=request.headers.get("X-Test-User")))

        def login_required(handler):
            @functools.wraps(handler)
            async def wrapped(*args, **kwargs):
                if not request.headers.get("X-Test-User"):
                    return jsonify(code=401, message="请登录", data=None), 401
                return await handler(*args, **kwargs)

            return wrapped

        apps.login_required = login_required
        utils = ModuleType("api.utils.api_utils")
        utils.get_json_result = lambda code=0, message="success", data=None: jsonify(code=code, message=message, data=data)

        async def getter():
            return self.host

        boundary = ModuleType("api.db.services.medical_assistant_service")
        boundary.get_medical_assistant_service = getter
        boundary.unavailable_session = service.unavailable_session
        path = ROOT / "api/apps/restful_apis/medical_assistant_api.py"
        spec = importlib.util.spec_from_file_location("medical_api_offline", path)
        module = importlib.util.module_from_spec(spec)
        module.manager = self.blueprint
        # Fully restored after import; subsequent suites see their real modules.
        with patch.dict("sys.modules", {"api.apps": apps, "api.utils.api_utils": utils, "api.db.services.medical_assistant_service": boundary}):
            spec.loader.exec_module(module)
        self.module = module
        self.app.register_blueprint(self.blueprint, url_prefix="/api/v1")
        self.client = self.app.test_client()
        self.headers = {"X-Test-User": "owner"}

    async def test_login_and_fixed_owner_required(self):
        path = "/api/v1/medical-assistant/session"
        self.assertEqual((await self.client.get(path)).status_code, 401)
        foreign = await self.client.get(path, headers={"X-Test-User": "foreign"})
        self.assertEqual(foreign.status_code, 403)
        self.assertIsNone((await foreign.get_json())["data"])
        self.assertEqual(self.policy.authorizations, 0)

    async def test_turn_request_returns_envelope_and_restores_session(self):
        rid = str(uuid4())
        response = await self.client.post("/api/v1/medical-assistant/turn", headers=self.headers, json={"request_id": rid, "question": "费用？"})
        body = await response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["code"], 0)
        self.assertEqual(body["data"]["turn"]["question"], "费用？")
        session = await (await self.client.get("/api/v1/medical-assistant/session", headers=self.headers)).get_json()
        self.assertEqual(session["data"]["turns"][0]["request_id"], rid)
        self.assertEqual(self.policy.authorizations, 2)
        clear = await self.client.post("/api/v1/medical-assistant/clear", headers=self.headers, json={"request_id": str(uuid4())})
        self.assertEqual((await clear.get_json())["data"]["turns"], [])

    async def test_browser_cannot_override_deployment_scope(self):
        for extra in ("tenant_id", "kb_id", "budget", "output_dir", "model"):
            response = await self.client.post("/api/v1/medical-assistant/turn", headers=self.headers, json={"request_id": str(uuid4()), "question": "费用？", extra: "attack"})
            self.assertEqual(response.status_code, 400)
        response = await self.client.get("/api/v1/medical-assistant/session?tenant_id=foreign", headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.policy.calls, [])

    async def test_missing_configuration_is_unavailable(self):
        async def unavailable():
            return None

        self.module.get_medical_assistant_service = unavailable
        response = await self.client.get("/api/v1/medical-assistant/session", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await response.get_json())["data"]["state"], "unavailable")
        response = await self.client.post("/api/v1/medical-assistant/turn", headers=self.headers, json={"request_id": str(uuid4()), "question": "费用？"})
        self.assertEqual(response.status_code, 503)

    async def test_internal_exception_body_never_reaches_browser(self):
        async def broken():
            raise RuntimeError("/private/path provider api_key=secret")

        self.module.get_medical_assistant_service = broken
        with self.assertLogs(level="ERROR"):
            response = await self.client.get("/api/v1/medical-assistant/session", headers=self.headers)
        self.assertNotIn("secret", await response.get_data(as_text=True))
        self.assertEqual((await response.get_json())["data"]["state"], "unavailable")


if __name__ == "__main__":
    unittest.main()
