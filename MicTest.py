import sounddevice as sd
print(sd.query_devices())

import sounddevice as sd
import time

# --- 設定 ---
DURATION = 1  # テスト録音の時間（秒）
SAMPLE_RATE = 16000

print("利用可能な入力デバイスのテストを開始します...")
print("-" * 30)

# 利用可能な全てのデバイスを調べる
for i, device in enumerate(sd.query_devices()):
    # 入力チャンネルが1つ以上あるデバイスのみをテスト対象とする
    if device['max_input_channels'] > 0:
        print(f"テスト中... デバイスID: {i}, 名前: {device['name']}")

        # 試すチャンネル数のリスト (まずは1チャンネルから試すのが一般的)
        channel_options = [1, 2, device['max_input_channels']]

        for ch in sorted(list(set(channel_options))):  # 重複を除いてソート
            try:
                print(f"  > チャンネル数: {ch} で試行中...", end="")

                # 実際に録音を試みる
                sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=ch, device=i, dtype='float32')
                sd.wait()

                print(" ✅ 成功！この組み合わせは利用可能です。")
                # 成功したら、これ以上このデバイスで試す必要はないのでループを抜ける
                break

            except Exception as e:
                # エラーが出たら失敗と表示
                print(f" ❌ 失敗 ({type(e).__name__})")
        print("-" * 30)

import pyaudio
import time
import numpy as np

# --- 設定 ---
# あなたの環境に合わせて、テストしたいデバイスの情報を設定
TARGET_MIC_NAME = "マイク配列"  # Windows内蔵マイクの一般的な名前の一部
CHANNELS = 1
RATE = 16000
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16

# --- グローバル変数 ---
# この変数がコールバック関数でインクリメントされるかを確認する
callback_counter = 0

# --- コールバック関数 ---
def test_callback(in_data, frame_count, time_info, status):
    """
    ストリームからデータが来るたびに呼び出される関数。
    呼び出された回数を数えるだけ。
    """
    global callback_counter
    callback_counter += 1
    # そのままデータを返す必要がある
    return (in_data, pyaudio.paContinue)


# --- メイン処理 ---
p = pyaudio.PyAudio()

# デバイスを名前で自動検出
input_device_index = None
print("利用可能なマイクを探しています...")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0 and TARGET_MIC_NAME in info['name']:
        input_device_index = i
        print(f"✅ マイク検出: index={i}, name={info['name']}")
        break

if input_device_index is None:
    print(f"❌ '{TARGET_MIC_NAME}' を含むマイクが見つかりませんでした。")
    p.terminate()
    exit()

# ストリームを開く
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=test_callback)

# ストリームを開始して10秒間待つ
print("\nストリームを開始します。10秒間、コールバックが呼ばれるかテストします...")
stream.start_stream()

for i in range(10):
    time.sleep(1)
    print(f"{i+1}秒経過... (コールバック呼び出し回数: {callback_counter})")

# ストリームを停止してリソースを解放
print("\nストリームを停止します。")
stream.stop_stream()
stream.close()
p.terminate()

print("テスト完了。")