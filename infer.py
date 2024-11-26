import re

import torch

from model import BertForWordBoundaryDetection

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

bert = BertForWordBoundaryDetection()
bert.load_state_dict(torch.load("pytorch_model.bin"))
bert.to(device)


def preprocess(text: str, digit_token: str = "<N>") -> list[str]:
    """
    Preprocess text for inference. This function removes all text that is not hangul, newlines, or whitespace and
    replaces all numbers with a `digit_token`. It then returns a list of all pairs of characters the precede and follow
    a newline.

    Parameters:
    - text (str): The text to preprocess
    - digit_token (str, optional): The digit token. Default: "<N>".

    Returns:
    - list[str]: All pairs of strings that precede and follow one or more newline characters.
    """
    # Remove all characters that are not hangul, digits, or newlines
    text = re.sub(r"[^\u3131-\uD79D\d\s\n]", "", text)
    # Replace all numbers with the digit token
    text = re.sub(r"\d+", digit_token, text)
    # Return all pairs of characters that precede and follow one or more newline characters
    return [
        re.sub(r"\n+", " ", nl)
        for nl in re.findall(r"[^\s(]+?\n+[^\s)]+", text)
    ]


def infer(input_text: str | list[str]) -> torch.Tensor:
    """
    Infer whether the characters of a string that precede and follow white space are of the same word (0) or different
    words (1).

    Parameters:
    - input_text (str | list[str]): A string or list of strings to run inference on.

    Returns:
    - torch.Tensor: The predictions made by the model.
    """
    inputs = bert.tokenize_function(input_text)
    outputs = bert(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    )
    predictions = (torch.sigmoid(outputs) > 0.5).int()
    return predictions


if __name__ == "__main__":
    text = """공장

훗날, 대극장을 설계한 건축가에 의해 처음 그 존재가 알려져
세상에 흔히 '붉은 벽돌의 여왕'으로 소개된 그 여자 벽돌공의 이
름은 춘희春姬이다. 전쟁이 끝나가던 해 겨울, 그녀는 한 거지 여자
에 의해 마구간에서 태어났다. 세상에 나왔을 때 이미 칠 킬로그
램에 달했던 그녀의 몸무게는 열네 살이 되기 전에 백 킬로그램을
넘어섰다. 벙어리였던 그녀는 자신만의 세계 안에 고립되어 외롭
게 자랐으며 의붓아버지인 文으로부터 벽돌 굽는 모든 방법을 배
웠다. 팔백여 명의 목숨을 앗아간 대화재 이후, 그녀는 방화범으로
체포되어 교도소에 수감되었다. 영어의 시간은 참혹했으며 그
녀는 오랜 교도소 생활 끝에 벽돌공장으로 돌아왔다. 당시 그녀의
1부 | 부두 9"""
    preprocessed = preprocess(text)
    predictions = infer(preprocessed)
    for string, prediction in zip(preprocessed, predictions):
        pred = prediction.item()
        print(f"input: {string}\tprediction: {pred} ({"different words" if pred else "same word"})")
