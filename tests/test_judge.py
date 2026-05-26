from rag_zh.judge import parse_judge_response


def test_parse_json_judge_response():
    result = parse_judge_response('{"correct": true, "rationale": "语义一致"}')

    assert result.correct is True
    assert result.rationale == "语义一致"


def test_parse_text_judge_response_fallback():
    result = parse_judge_response("不正确：回答缺少核心事实")

    assert result.correct is False
