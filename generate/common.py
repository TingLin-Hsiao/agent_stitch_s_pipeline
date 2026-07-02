# -*- coding: utf-8 -*-
import csv
import json
import os
import re
from pathlib import Path

def load_prompt_file(filename):
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt_file("generate_stitch_s.txt")

def compact_observation(text, max_chars=2400):
    """
    針對過長的工具觀測結果進行截斷，保留頭部與尾部的關鍵資訊。
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    
    # 頭部保留 1400 字元，尾部保留 800 字元
    head = text[:1400]
    tail = text[-800:]
    return head + "\n...[TRUNCATED]...\n" + tail


def parse_json_maybe(value, fallback=None):
    if fallback is None:
        fallback = value
    if isinstance(value, str):
        try:
            return json.loads(value) if value else fallback
        except Exception:
            return fallback
    return value if value is not None else fallback


def dumps_json_field(value, empty_value=None):
    if value is None:
        value = [] if empty_value is None else empty_value
    return json.dumps(value, ensure_ascii=False)


def loads_json_field(value, fallback=None):
    if fallback is None:
        fallback = []
    if isinstance(value, str):
        try:
            return json.loads(value) if value else fallback
        except Exception:
            return fallback
    return value if value is not None else fallback


def normalize_text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def strip_json_fence(text):
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def clean_say_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"</?SAY>", "", text).strip()
    return text


def clean_private_state(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("[SOPR]", "").replace("[EOPR]", "")
    return text.strip()


def normalize_available_tools(tools):
    """
    將 OpenAI-style tools schema 壓成目標輸出格式:
    [{"name": "...", "description": "..."}]
    """
    tools = parse_json_maybe(tools, [])
    if not isinstance(tools, list):
        return []

    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        normalized.append({
            "name": name,
            "description": fn.get("description", "") if isinstance(fn, dict) else "",
        })
    return normalized


def normalize_tool_call_for_output(tool_call):
    """
    將各來源的 tool_call 統一成:
    {"function": {"name": "...", "arguments": {...}}}
    """
    if not isinstance(tool_call, dict):
        return {"function": {"name": "unknown", "arguments": {}}}

    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    name = fn.get("name") or tool_call.get("name") or "unknown"
    arguments = fn.get("arguments", tool_call.get("arguments", {}))
    arguments = parse_json_maybe(arguments, arguments)
    if arguments is None:
        arguments = {}
    return {"function": {"name": name, "arguments": arguments}}


def normalize_tool_result_for_output(tool_name, tool_result):
    """
    將 tool result 統一成 JSON object，避免輸出成裸字串。
    """
    parsed = parse_json_maybe(tool_result, None)
    if isinstance(parsed, dict):
        if "name" not in parsed:
            parsed = {"name": tool_name, **parsed}
        return parsed
    return {"name": tool_name, "response": "" if tool_result is None else str(tool_result)}


def tool_call_line(tool_call):
    return f"<TOOL_CALL>{json.dumps(tool_call, ensure_ascii=False)}</TOOL_CALL>"


def tool_result_line(tool_result):
    return f"<TOOL_RESULT>{json.dumps(tool_result, ensure_ascii=False)}</TOOL_RESULT>"


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def pop_pending_tool_call(pending_tool_calls, pending_tool_calls_by_id, tool_call_id=None):
    # Prefer explicit tool_call_id when present; otherwise consume calls in order.
    if tool_call_id and tool_call_id in pending_tool_calls_by_id:
        tool_call = pending_tool_calls_by_id.pop(tool_call_id)
        pending_tool_calls = [
            tc for tc in pending_tool_calls
            if not (isinstance(tc, dict) and tc.get("id") == tool_call_id)
        ]
        return tool_call, pending_tool_calls, pending_tool_calls_by_id

    if pending_tool_calls:
        tool_call = pending_tool_calls.pop(0)
        if isinstance(tool_call, dict) and tool_call.get("id"):
            pending_tool_calls_by_id.pop(tool_call.get("id"), None)
        return tool_call, pending_tool_calls, pending_tool_calls_by_id

    return {}, pending_tool_calls, pending_tool_calls_by_id


def get_tool_response_call_id(tool_response):
    if not isinstance(tool_response, dict):
        return None
    return (
        tool_response.get("tool_call_id")
        or tool_response.get("call_id")
        or tool_response.get("id")
    )


def append_tool_step(canonical, step_id, tool_call, tool_result):
    output_tool_call = normalize_tool_call_for_output(tool_call)
    tool_name = get_tool_name(output_tool_call)
    output_tool_result = normalize_tool_result_for_output(tool_name, tool_result)
    call_line = tool_call_line(output_tool_call)
    result_line = tool_result_line(output_tool_result)

    canonical["steps_full"].append({
        "step_id": step_id,
        "tool_call": output_tool_call,
        "tool_result": output_tool_result
    })

    canonical["tool_steps"].append({
        "order": step_id + 1,
        "tool_call_line": call_line,
        "tool_result_line": result_line,
    })

    canonical["steps_compact"].append({
        "step_id": step_id,
        "tool_name": tool_name,
        "tool_call_brief": json.dumps(output_tool_call, ensure_ascii=False)[:300],
        "observation_brief": compact_observation(json.dumps(output_tool_result, ensure_ascii=False))
    })


def canonicalize_row(row):
    """
    將 voidful/agent-sft 格式的原始資料統一轉換為 Canonical Row 格式。
    
    格式如下。巢狀欄位在回傳前會序列化成 JSON 字串，避免 Ray/Arrow
    對不同 tool arguments schema 做 merge 時失敗。
    
    原始 canonical 格式:
    {
      "id": str,
      "source": str,
      "user_request": str,
      "steps_full": list,     # 完整步驟，包含完整 observation，組裝用 (不進 Prompt)
      "steps_compact": list,  # 壓縮步驟，只包含簡短說明，推論用 (進 Prompt)
      "available_tools": list,
      "tool_steps": list,
      "final_answer_hint": str
    }
    """
    rec_id = row.get("id") or "unknown_id"
    source = row.get("source") or "unknown_source"
    available_tools = normalize_available_tools(row.get("tools"))
    
    msgs = row.get("messages")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs) if msgs else []
        except:
            msgs = []
    elif msgs is None:
        msgs = []
            
    canonical = {
        "id": rec_id,
        "source": source,
        "user_request": "",
        "steps_full": [],
        "steps_compact": [],
        "available_tools": available_tools,
        "tool_steps": [],
        "final_answer_hint": ""
    }
    
    step_id = 0
    pending_tool_calls = []
    pending_tool_calls_by_id = {}
    
    for msg in msgs:
        role = msg.get("role")
        raw_content = msg.get("content", "")
        content = normalize_text_content(raw_content)
        
        if role == "user":
            canonical["user_request"] = content
        elif role == "assistant":
            # 取得 tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tool_calls = as_list(tool_calls)
                pending_tool_calls = list(tool_calls)
                pending_tool_calls_by_id = {
                    tc.get("id"): tc
                    for tc in pending_tool_calls
                    if isinstance(tc, dict) and tc.get("id")
                }
                tool_responses = msg.get("tool_responses")
                if tool_responses:
                    for tool_call, tool_response in zip(tool_calls, as_list(tool_responses)):
                        append_tool_step(canonical, step_id, tool_call, tool_response)
                        step_id += 1
                    pending_tool_calls = pending_tool_calls[len(as_list(tool_responses)):]
                    pending_tool_calls_by_id = {
                        tc.get("id"): tc
                        for tc in pending_tool_calls
                        if isinstance(tc, dict) and tc.get("id")
                    }
            elif content:
                canonical["final_answer_hint"] = content
        elif role == "tool":
            # 工具結果。voidful/agent-sft 常把結果放在 tool_responses list，
            # 且 content 可能是 null。
            tool_responses = as_list(msg.get("tool_responses")) if msg.get("tool_responses") else [content]
            tool_call_id = msg.get("tool_call_id")

            for tool_result in tool_responses:
                response_tool_call_id = get_tool_response_call_id(tool_result) or tool_call_id
                # 用 tool_call_id 對齊對應的呼叫；若資料缺少 id，退回 FIFO。
                tool_call, pending_tool_calls, pending_tool_calls_by_id = pop_pending_tool_call(
                    pending_tool_calls,
                    pending_tool_calls_by_id,
                    response_tool_call_id,
                )
                append_tool_step(canonical, step_id, tool_call, tool_result)
                step_id += 1
            
    if msgs and msgs[-1].get("role") == "assistant" and not canonical["final_answer_hint"]:
        canonical["final_answer_hint"] = normalize_text_content(msgs[-1].get("content", ""))
        
    return {
        "id": canonical["id"],
        "source": canonical["source"],
        "user_request": canonical["user_request"],
        "steps_full_json": dumps_json_field(canonical["steps_full"]),
        "steps_compact_json": dumps_json_field(canonical["steps_compact"]),
        "available_tools_json": dumps_json_field(canonical["available_tools"]),
        "tool_steps_json": dumps_json_field(canonical["tool_steps"]),
        "final_answer_hint": canonical["final_answer_hint"],
        "language": canonical.get("language", "zh-TW"),
        "context_json": dumps_json_field(canonical.get("context", [])),
    }


def tool_reference_filter_failure(row):
    steps_full = loads_json_field(row.get("steps_full_json"), [])
    reference_answer = (row.get("final_answer_hint") or "").strip()
    reasons = []
    if not steps_full:
        reasons.append("missing_tool_call_tool_result")
    if not reference_answer:
        reasons.append("missing_reference_answer")
    if not reasons:
        return None
    return {
        "id": row.get("id"),
        "status": "dropped",
        "drop_reason": "+".join(reasons),
        "raw_patch": None,
        "error": "Filtered before inference because tool call/result or reference answer is missing.",
        "sft_data": None,
    }


def has_tool_call_tool_result_and_reference(row):
    return tool_reference_filter_failure(row) is None


def get_tool_name(tool_call):
    if not isinstance(tool_call, dict):
        return "unknown"
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        return fn.get("name") or "unknown"
    return tool_call.get("name") or "unknown"

def build_user_payload(row):
    """
    從 canonical 欄位組裝給 Gemma 的 User Payload
    """
    available_tools = loads_json_field(row.get("available_tools_json"), [])
    steps_compact = loads_json_field(row.get("steps_compact_json"), [])
    payload = {
        "id": row["id"],
        "source": row["source"],
        "user": row["user_request"],
        "available_tools": available_tools,
        # Prompt only needs compact observations to write SAY/SOPR patches.
        # Full tool call/result lines stay in tool_steps / steps_full for deterministic assembly.
        "tool_steps": steps_compact,
        "reference_answer": row.get("final_answer_hint", ""),
        "language": row.get("language", "zh-TW"),
    }
    return json.dumps(payload, ensure_ascii=False)

def assemble_agent_stitch_s(canonical_row, raw_patch_str):
    """
    將 Gemma 生成的 Patch 與原始 Trajectory 進行確定性組裝 (Deterministic Assembly)。
    """
    try:
        patch = json.loads(strip_json_fence(raw_patch_str))
    except Exception as e:
        # JSON 解析失敗，返回空或標記錯誤
        return {
            "error": f"JSON parse error: {str(e)}",
            "drop_reason": "patch_json_parse_failed",
            "linear_target": None,
        }
    
    if patch.get("drop_reason"):
        return {
            "drop_reason": patch["drop_reason"],
            "linear_target": None,
        }
    
    parts = []
    
    # 1. first_say
    first_say = clean_say_text(patch.get("first_say", ""))
    parts.append(f"<SAY>{first_say}</SAY>")
    
    # 2. 中間步驟
    steps_patch = patch.get("steps", [])
    steps_full = canonical_row.get("steps_full", [])
    
    # 確保步驟數量對齊，若不對齊則捨棄
    if len(steps_patch) != len(steps_full):
        return {
            "drop_reason": f"step_count_mismatch: patch={len(steps_patch)}, full={len(steps_full)}",
            "linear_target": None,
        }
    
    for step_patch, step in zip(steps_patch, steps_full):
        pre_tool_private = clean_private_state(step_patch.get("pre_tool_private_state", ""))
        post_tool_say = clean_say_text(step_patch.get("post_tool_say", ""))
        
        # 寫入私有思維狀態 [SOPR]...[EOPR]
        parts.append(f"[SOPR]{pre_tool_private}[EOPR]")
        
        # 插入原始 Tool Call，接著放工具執行期間說給使用者聽的 bridge SAY。
        tool_call_str = json.dumps(step["tool_call"], ensure_ascii=False)
        parts.append(f"<TOOL_CALL>{tool_call_str}</TOOL_CALL>")
        parts.append(f"<SAY>{post_tool_say}</SAY>")

        # 最後插入原始 Tool Result。這樣順序會對齊目標格式:
        # TOOL_CALL -> SAY while tool is running -> TOOL_RESULT。
        tool_result_str = json.dumps(step["tool_result"], ensure_ascii=False)
        parts.append(f"<TOOL_RESULT>{tool_result_str}</TOOL_RESULT>")
        
    # 3. final_private_state
    final_private = clean_private_state(patch.get("final_private_state", ""))
    parts.append(f"[SOPR]{final_private}[EOPR]\n[EOR]")
    
    # 4. final_say
    final_say = clean_say_text(patch.get("final_say", ""))
    parts.append(f"<SAY>{final_say}</SAY>")
    
    linear_target = "\n\n".join(parts)
    
    return {
        "drop_reason": None,
        "linear_target": linear_target,
        "translated_user": patch.get("translated_user")
    }


def build_output_input(canonical_row):
    return {
        "language": canonical_row.get("language", "zh-TW"),
        "context": canonical_row.get("context", []),
        "user": canonical_row["user_request"],
        "available_tools": canonical_row.get("available_tools", []),
        "tool_steps": canonical_row.get("tool_steps", []),
        "reference_answer": canonical_row.get("final_answer_hint", ""),
    }


def generate_sft_row(canonical_row, assembled_result):
    """
    產生最後用於訓練的 SFT 樣本結構
    """
    return {
        "id": canonical_row["id"],
        "source": canonical_row["source"],
        "user": assembled_result.get("translated_user") or canonical_row["user_request"],
        "msg": assembled_result["linear_target"],
        "input": build_output_input(canonical_row)
    }


def load_only_ids(path):
    if not path:
        return None

    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line_no == 1 and line.lower() == "id":
                continue
            if line.startswith("{"):
                row = json.loads(line)
                row_id = row.get("id")
            else:
                row_id = line.split(",", 1)[0]
                if line_no == 1 and row_id == "id":
                    continue
            if row_id:
                ids.add(row_id)
    return ids
