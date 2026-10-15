"""Authenticated, fixed-scope Shanghai medical policy assistant HTTP entry."""

import logging

from quart import request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from api.apps import current_user, login_required
from api.db.services.medical_assistant_service import get_medical_assistant_service, unavailable_session
from api.utils.api_utils import get_json_result
from common.constants import RetCode


_MESSAGES = {
    "invalid_question": "请输入有效的问题（最多 10000 字）。",
    "invalid_request_id": "请求标识无效，请刷新页面确认当前状态。",
    "request_id_body_mismatch": "该请求标识已用于另一条指令。",
    "assistant_running": "当前问题正在执行，请等待结果。",
    "unresolved_provider_usage": "服务用量待核对，暂不可查询。",
    "usage_reconciliation_invalid": "服务用量记录待核对，暂不可查询。",
    "session_token_budget_exhausted": "本运行会话额度已用完，暂不可查询。",
    "command_unconfirmed": "上一条指令尚未确认结束，暂不可查询。",
    "source_unavailable": "政策资料尚未就绪，暂不可查询。",
}


async def _perform(action):
    try:
        host = await get_medical_assistant_service()
    except Exception:
        logging.exception("Medical assistant deployment unavailable")
        host = None
    if host is None:
        if action == "session":
            return get_json_result(data=unavailable_session())
        return get_json_result(code=RetCode.OPERATING_ERROR, message="服务暂未配置，暂不可查询。"), 503
    try:
        await host.authorize(current_user.id)
        if request.args:
            raise ValueError("invalid_parameters")
        if action == "session":
            return get_json_result(data=await host.snapshot())
        body = await request.get_json()
        required = {"request_id", "question"} if action == "turn" else {"request_id"}
        if not isinstance(body, dict) or set(body) != required:
            raise ValueError("invalid_parameters")
        if action == "turn":
            data = await host.submit(body["request_id"], body["question"])
        else:
            data = await host.clear(body["request_id"])
        return get_json_result(data=data)
    except PermissionError:
        return get_json_result(code=RetCode.PERMISSION_ERROR, message="当前账户无权访问此助手会话。"), 403
    except (BadRequest, UnsupportedMediaType):
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="请求必须包含有效的 JSON 参数。"), 400
    except ValueError as exc:
        reason = str(exc)
        parameter = reason in {"invalid_parameters", "invalid_question", "invalid_request_id", "request_id_body_mismatch"}
        return get_json_result(
            code=RetCode.ARGUMENT_ERROR if parameter else RetCode.OPERATING_ERROR, message=_MESSAGES.get(reason, "请求参数无效。" if parameter else "服务暂不可查询，请确认当前状态。")
        ), 400 if parameter else 409
    except Exception:
        logging.exception("Medical assistant request failed")
        return get_json_result(code=RetCode.EXCEPTION_ERROR, message="服务暂不可查询，请稍后读取会话状态。"), 500


@manager.route("/medical-assistant/session", methods=["GET"])  # noqa: F821
@login_required
async def medical_assistant_session():
    return await _perform("session")


@manager.route("/medical-assistant/turn", methods=["POST"])  # noqa: F821
@login_required
async def medical_assistant_turn():
    return await _perform("turn")


@manager.route("/medical-assistant/clear", methods=["POST"])  # noqa: F821
@login_required
async def medical_assistant_clear():
    return await _perform("clear")
