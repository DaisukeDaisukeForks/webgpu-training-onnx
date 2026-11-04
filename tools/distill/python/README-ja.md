# 学習済みRTMW3Dを引き継いで37点モデルへ圧縮
公式MMPoseのモデル定義と公式RTMW3D-Xの`.pth`を直接読み込み、構造化プルーニングでチャネルを削る。新しい小型CNNをランダム初期化する処理はない。圧縮自体には画像、動画、関節の正解データを要求しない。元の学習済みテンソルを切り出して保存する。
## Windows・CUDA 11.8
Python 3.10、Git、対応するNVIDIAドライバーを用意する。リポジトリのルートで実行する。インストールはプロジェクト専用の`.venv-prune`内へ行う。PyTorchのcu118版と対応するMMCVの配布済みwheelを使い、CUDA拡張のローカルビルドは行わない。
```powershell
powershell -ExecutionPolicy Bypass -File tools/distill/python/setup-windows.ps1
```
`GPU available True`なら追加学習にCUDAを使える。圧縮処理はCPUで実行する。既存のPython環境や既存の教師データキャッシュは削除しない。
## 画像なしで重みを圧縮
```powershell
.\.venv-prune\Scripts\python.exe tools/distill/python/compress.py --ratio 0.5 --output pruned37-output
```
公式配布先の重みを自動取得し、SHA-256を確認する。既に取得済みなら`--checkpoint "重みのパス.pth"`を指定できる。`--ratio 0.5`は対象チャネルの削減率であり、精度保持率や速度倍率を意味しない。チャネルを実際に削って密な演算のサイズを減らすため、重みをゼロにするだけの疎行列化ではない。
出力は`compressed.pt`、`pruned37.onnx`、`pruned37.schema.json`、`compression.json`。パラメータ数と推定MAC数の削減前後をJSONへ記録する。GPU実測速度や認識精度を測った結果とは区別する。
さらに小さくする場合は、学習済みヘッドの行列をSVDで低ランク分解する。分解後の重みも元の重みから計算し、ランダム初期化しない。入出力の関節数と空間寸法は変えない。
```powershell
.\.venv-prune\Scripts\python.exe tools/distill/python/compress.py --ratio 0.75 --head-rank 0.25 --output pruned37-small
```
公式重みで実行した結果、標準設定は92,342,447→40,057,519パラメータ、29.31G→12.95G MAC。上記の小型設定は12,785,215パラメータ、3.80G MACとなり、37点ONNXを生成できた。これらは構造・演算量の結果であり、元モデルと同等精度を確認した結果ではない。小型設定では特に精度回復が必要になり得る。
出力は身体17点、足6点、手14点の計37点。足は左右のかかと・親指側・小指側、手は左右5指先・人差し指付け根・小指付け根を含む。既存の関節間Attentionの情報を保つため、内部の133トークンは維持して最後に37点を取り出す。37点への出力削減だけを高速化として数えず、CNN側の構造化圧縮を行う。
ONNX入力はRGB 0〜1、`[1,3,384,288]`。正規化はモデル内部で行う。出力`keypoints`は`[1,37,3]`、XYは切り抜き内288×384ピクセル、Zは骨盤基準メートル。XYとZは異なる単位なので、そのまま同一空間のXYZとしてOSCへ送らない。
## 圧縮後の精度を回復
以下は圧縮済みの学習済み重みから始める追加学習。元のRTMW3Dを固定した教師とし、両モデルのXYZ分布と脚・足先の相対ベクトルを合わせる。データセットの正解座標による再学習はしない。新しいランダム初期化モデルも作らない。
画像・動画は、圧縮で変わった応答を元モデルへ近づける入力としてだけ必要。画像なしの圧縮と、入力を使った精度回復は別コマンドにしている。学習用と評価用は別の撮影を使う。
```powershell
.\.venv-prune\Scripts\python.exe tools/distill/python/train.py prepare --input "C:\training\videos" "C:\training\images" --output distill-data --fps 4
.\.venv-prune\Scripts\python.exe tools/distill/python/train.py train --manifest distill-data/images.json --model pruned37-output/compressed.pt --device cuda --epochs 10 --batch 4 --output pruned37-finetuned
```
全身が入る人物枠を`prepare --roi x,y,幅,高さ`で0〜1指定できる。省略時は画像全体を使う。複数人物の映像は対象人物を切り抜いて渡す。抽出フレームと入力リストはローカル保存し、画像や動画をサーバーへ送らない。
60秒ごとに`latest.pt`へ重み・構造・Adam状態・乱数・順序・学習位置・累積時間を原子的に保存する。30分ごとに別名バックアップも残す。Ctrl+Cは現在の更新後に保存して停止する。`result.json`の`training_hours`は小数1桁で、10000時間以上も表せる。
```powershell
.\.venv-prune\Scripts\python.exe tools/distill/python/train.py train --manifest distill-data/images.json --resume pruned37-finetuned/latest.pt --device cuda --epochs 30 --batch 2 --output pruned37-finetuned
```
明示したエポック・バッチ・学習率は保存ファイルより優先する。指定しなければ保存設定を引き継ぐ。元のブラウザ版WB73チェックポイントとはモデル構造が違うため、このPython圧縮モデルの復元には使わない。
## 公式配布元・ライセンス
MMPose: https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9 — Apache-2.0。
元の重み: https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth
SHA-256: `b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca`。
Torch-Pruning: https://github.com/VainF/Torch-Pruning — MIT。PyTorch: https://github.com/pytorch/pytorch — BSD-style。
公式モデルは登録済みのMMPoseクラスをそのまま使用し、重みの欠損・余剰があれば停止する。Hugging FaceのONNXを教師入力にしない。
## 実行確認の範囲
公式の837テンソルを欠損・余剰なしで読み込み、上記2設定の画像なし圧縮とONNX保存をCPUで実行した。標準設定の圧縮モデルで、実映像1フレームを使った更新・保存・再開・手入力した学習率の優先まで実行した。これは長時間の精度回復学習でも、認識精度の達成でもない。CUDAでの速度とWindowsのセットアップ実行は未確認。Windows用MMCVの対応wheelが配布されていることは確認済み。
圧縮直後の実映像1フレームでは、元教師に対する37点の平均XY差が標準設定で約162.5px、小型設定で約182.1px（288×384切り抜き内）となった。この設定では認識精度を保てておらず、出力は精度回復学習の開始用チェックポイントとして扱う。小型化できたことだけを実用モデルの完成とはしない。教師への一致度は実測正解に対する精度でもない。
