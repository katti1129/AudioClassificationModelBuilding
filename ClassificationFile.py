import os
import shutil
from pathlib import Path

# --- 設定項目 ---

# 1. 整理したい音声ファイルが入っているフォルダのパス
SOURCE_FOLDER = "./data/ESC-50-master/audio_trimmed/"

# 2. ★★★クラス番号と、それぞれを格納したいフォルダのパスを指定★★★
#    'クラス番号': '格納先のフォルダパス' の形式で追加できます。
TARGET_FOLDERS = {
    "42": "./data/ESC-50-master/organized/keyboard_typing_42",
    "25": "./data/ESC-50-master/organized/laughing_25"
}


def organize_files_into_separate_folders():
    """
    TARGET_FOLDERSで指定されたクラスの音声ファイルを、
    それぞれ対応する専用のフォルダに移動します。
    """
    print(f"処理を開始します...")
    print(f"読み込み元: {SOURCE_FOLDER}")
    print("--- 整理対象 ---")
    for class_id, folder_path in TARGET_FOLDERS.items():
        print(f"クラス '{class_id}' -> フォルダ '{folder_path}'")
    print("----------------")

    source_path = Path(SOURCE_FOLDER)

    # 処理したファイル数をカウント
    file_count = 0
    skipped_count = 0

    # フォルダ内の全てのファイルをループ
    for file_path in source_path.glob('*.wav'):  # .wavファイルのみを対象
        filename = file_path.name

        try:
            # ファイル名を分解して末尾の番号（ターゲットID）を取得
            parts = filename.split('-')
            target_id = parts[-1].split('.')[0]

            # ターゲットIDが指定された辞書のキーに含まれているかチェック
            if target_id in TARGET_FOLDERS:
                # 対応する保存先フォルダのパスを取得
                destination_folder_path = Path(TARGET_FOLDERS[target_id])

                # そのフォルダが存在しない場合は作成
                destination_folder_path.mkdir(parents=True, exist_ok=True)

                # ファイルを新しいフォルダに移動
                shutil.move(file_path, destination_folder_path / filename)

                print(f"移動: {filename} -> フォルダ '{destination_folder_path.name}'")
                file_count += 1
            else:
                # 対象外のファイルはスキップ
                skipped_count += 1

        except IndexError:
            print(f"スキップ (命名規則外): {filename}")
        except Exception as e:
            print(f"エラー: {filename} の処理中に問題が発生しました - {e}")

    print(f"\n処理が完了しました。")
    print(f"合計 {file_count} 個のファイルを整理しました。")
    if skipped_count > 0:
        print(f"{skipped_count} 個のファイルは対象外のためスキップしました。")


if __name__ == "__main__":
    # **重要**: 本番のデータで実行する前に、
    # 少数のファイルをコピーしたテスト用のフォルダでスクリプトが正しく動作するか確認してください。
    organize_files_into_separate_folders()