"""Privacy-gated partial reranking; no benchmark labels or network access."""

import math
import re


def eligible_source(source_id, document, blocked):
    title = document.get('文件名称', '')
    return source_id not in blocked and document.get('是否公开发布') == '是' and bool(title) and not re.fullmatch(r'[0-9a-fA-F-]{28,}', title) and not re.search(r'招聘|表彰|获奖|公示|送达|劳动关系|答复|协办|巡视|社会监督员队伍|宣传活动', title)


def sanitize_outbound(content):
    sensitive = r'联\s*系\s*人|联系.{0,5}(?:电话|地址)|电子邮箱|传真|(?<!\d)(?:1[3-9]\d{9}|\d{17}[\dXx])(?!\d)|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}'
    lines = [line for line in content.splitlines() if not re.search(sensitive, line)]
    text = '\n'.join(lines)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '[图片链接已省略]', text)
    text = re.sub(r'https?://[^\s)）]+', '[网页链接已省略]', text)
    text = re.sub(r'(?<!\d)0\d{2,3}[-— ]?\d{7,8}(?!\d)', '[固定电话已省略]', text)
    return text[:1800].strip()


def validate_scores(results, count):
    if count < 1 or len(results) != count:
        raise ValueError('Reranker must return every sent candidate')
    scores = {}
    for item in results:
        index, value = item.get('index'), item.get('relevance_score')
        if type(index) is not int or not 0 <= index < count or index in scores or type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid reranker index or score')
        scores[index] = value
    return [scores[i] for i in range(count)]


def reorder_slots(ids, slots, scores):
    if len(ids) != len(set(ids)) or slots != sorted(set(slots)) or any(type(s) is not int or not 0 <= s < min(32, len(ids)) for s in slots) or len(scores) != len(slots) or any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid fixed-slot rerank inputs')
    result = list(ids)
    order = sorted(range(len(slots)), key=lambda i: (-scores[i], slots[i]))
    for destination, source in zip(slots, order):
        result[destination] = ids[slots[source]]
    return result
