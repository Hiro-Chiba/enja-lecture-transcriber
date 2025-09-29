# 英日リアルタイム翻訳アプリ

Python と Tkinter を用いた英語音声→日本語テキスト翻訳アプリです。マイクから取得した英語音声を Google Speech Recognition API で文字起こしし、Google Cloud Translation API（利用可能な場合）で日本語へ翻訳します。認証情報が未設定の場合は `googletrans` ライブラリを用いた翻訳に自動的にフォールバックします。

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

### Google Cloud Translation API を利用する場合

1. Google Cloud プロジェクトで Translation API を有効化します。
2. サービスアカウントキー（JSON）をダウンロードするか、`gcloud auth application-default login` で認証情報を設定します。
3. サービスアカウントキーを利用する場合は `GOOGLE_APPLICATION_CREDENTIALS` 環境変数に JSON ファイルのパスを設定するか、`GOOGLE_API_KEY` に API キーを設定します。

アプリ側で API キーや認証情報をハードコードする必要はなく、上記の設定が完了していれば起動時に自動検出されます。

## 使い方

```bash
python app.py
```

1. 「開始」を押すとマイク入力の待機を開始します。
2. 話し終えると英語の認識結果と日本語訳が順に表示されます。
3. 翻訳を止めたい場合は「停止」を押してください。

## 精度向上のための工夫

- 音声の RMS 値を評価し、話者の声が遠かったり小さすぎる場合は自動で再キャリブレーションを行います。
- 英語のアクセント差を吸収するため、複数の言語コードで候補を取得し、信頼度が最も高い結果を選択します。
- 無音や認識失敗が続く場合はバックグラウンドノイズのキャリブレーション時間を延長し、次回以降の誤検出を抑えます。
- WebRTC ベースの音声区間検出 (VAD) で発話区間をミリ秒単位で切り出し、長時間の話でも安定した文字起こし精度を実現します。
- 公式の Google Cloud Translation API に対応し、安定した翻訳品質を確保しつつ、利用できない環境では自動的に `googletrans` へフォールバックします。

## 注意事項

- Google Speech Recognition API を利用するため、短時間に大量のリクエストを行うと制限がかかる可能性があります。
- 翻訳結果は機械翻訳によるものなので、内容を保証するものではありません。
