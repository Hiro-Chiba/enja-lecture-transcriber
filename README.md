# 英日リアルタイム翻訳アプリ

Python と Tkinter を用いた英語音声→日本語テキスト翻訳アプリです。マイクから取得した英語音声を Google Speech Recognition API で文字起こしし、`googletrans` ライブラリを用いて日本語に翻訳します。

## 必要環境

- Python 3.10 以上
- マイク入力が可能な環境
- インターネット接続（音声認識および翻訳 API 利用のため）

## インストール

```bash
python -m venv .venv
source .venv/bin/activate  # Windows の場合は .venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

```bash
python app.py
```

1. 「開始」を押すとマイク入力の待機を開始します。
2. 話し終えると英語の認識結果と日本語訳が順に表示されます。
3. 翻訳を止めたい場合は「停止」を押してください。

## 注意事項

- Google Speech Recognition API を利用するため、短時間に大量のリクエストを行うと制限がかかる可能性があります。
- 翻訳結果は機械翻訳によるものなので、内容を保証するものではありません。
