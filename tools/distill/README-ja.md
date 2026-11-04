# RTMW3D → Distill37 / WebGPU蒸留
[元の学習済み重みを引き継いで圧縮するPython版](python/README-ja.md)を追加した。公式の`.pth`を読み込み、画像なしの構造化プルーニングと低ランク分解を行う。必要な精度回復は圧縮済み重みから開始する。
公式RTMW3Dの学習済み重みを教師にして、別の軽量モデルをブラウザで学習する。教師の重みを35点へ切り詰める処理ではない。学習対象は37点、学生モデルは1,039,909パラメータ。教師生成・逆伝播・Adam・軽量モデル推論はWebGPUで実行する。GitHub Actionsは追加しない。
## 起動
公開ページの `/distill.html` を開く。元のWB73学習ページとチェックポイントは変更しない。
推論ランタイムはjsDelivrの `onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs` を使用し、WASMローダーとWASM本体も同じバージョンへ固定する。Hugging Faceの `text/plain` 配信URLをJavaScriptモジュールとして直接importしない。プレビューは専用ID `distill-preview` を使い、既存ページの全画面 `#overlay` と分離する。
既に持っている `rtmw3d-x-fp32.onnx` または `rtmw3d-l-fp32.onnx` を選び「WebGPUを準備」を押す。RabbitDaisuke/rtmw3d-webgpu-onnx の133点版は、内部の `simcc_x/y/z` をブラウザ内で出力へ戻すため、Pythonなしで蒸留できる。変換で重みは変更しない。FP32教師を使用する。出力名・形状の異なるONNXは受け付けない。
公式 `.pth` から新規に教師ONNXを作る場合は、次節の一度だけの変換を行う。教師を変更すると保存分布との整合が崩れるため、チェックポイントと異なるSHA-256の教師は混ぜない。
## 公式のモデル定義・重みから変換
上流: https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose3d
公式重みの配布元: https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/
RTMW3D-Xの重みは実際にダウンロードし、SHA-256 `b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca` を確認した。Lは公式ファイル名のハッシュ接頭辞 `794dbc78` を照合する。
以下はPython 3.10の環境で行う。WindowsではWSL2/Ubuntuでの変換を推奨。Pythonは教師の形式変換だけに使用し、蒸留学習には使用しない。
```bash
python3.10 -m venv .venv-teacher
source .venv-teacher/bin/activate
python -m pip install pip==24.0 setuptools==69.5.1 wheel
python -m pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r tools/distill/requirements-teacher.txt
python -m pip install --only-binary=:all: mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
git clone https://github.com/open-mmlab/mmpose.git teacher-upstream
git -C teacher-upstream checkout 759b39c13fea6ba094afc1fa932f51dc1b11cbf9
python -m pip install --no-deps --no-build-isolation -e ./teacher-upstream
python tools/distill/export_teacher.py --model x --upstream teacher-upstream --output teacher
```
`teacher/rtmw3d-x-raw-fp32.onnx` をブラウザで選ぶ。既存の公式.pthがある場合は `--checkpoint /path/to/checkpoint.pth` を渡せる。Lを教師にする場合は `--model l`。公式のConfigと登録済みモデルをそのまま使用し、未ロードの重みがあれば停止する。`.pth` は公式ハッシュ照合後にのみロードする。
## 何を蒸留するか
身体17点、足6点、手14点の計37点。手は左右それぞれ5指先・人差し指付け根・小指付け根。順序は `public/distill/schema.mjs`。
教師の入力はRGB 384×288、平均と標準偏差は公式値。学生は同じ人物領域のRGB 256×192、0〜1。人物枠を1.25倍へ拡張して縦横比0.75へ合わせ、領域外を黒で埋める。
教師のSimCC分布へ公式学習と同じbeta=10、温度2を適用し、隣接3ビンを合計する。学生の分布はX192・Y256・Z192ビン。分布はuint16へ量子化して保存し、使用時に再正規化する。1例あたり分布は47,360バイト、これに切り抜きJPEG・133点の座標・スコア・正解2Dラベルが加わる。
学生は拡張畳み込み→depthwise→pointwiseの残差CNNと、位置を保持した関節トークン・画像全体の特徴を使う。XYZを同じ画像特徴から出力するため、奥行き側にも足裏・脚の外観が入る。2D座標だけを別の3D推定器へ渡す構造ではない。
損失は教師分布とのKL、教師座標とのSmoothL1、脚・かかと・親指側・小指側を結ぶ相対ベクトル差、既存COCO/CMUの正解2DとのSmoothL1。足グループに独立した分母と重み2を与え、他部位の平均に埋もれさせない。奥行きの相対ベクトル差にも重みを付ける。足を床に固定する処理、脚の交差を禁止する処理、動かないほど良いという損失は入れない。
推論の座標は分布の最大ビンから復号する。複数ピークの中間へ座標を平均する方式を推論出力には使用しない。
## 教師データを作る
「既存画像から教師を作る」は既存のCOCO 40,000例＋CMU Foot 14,614例を利用する。同じ画像・人物枠・教師SHAなら再推論しない。既存画像と教師データのキャッシュ名・キー・内容は変更せず、蒸留ラベルを別のIndexedDBへ追加する。
動画はブラウザでファイルを選ぶ。人物枠は映像全体に対する `x,y,幅,高さ` を0〜1で指定する。動く足を含む範囲を指定し、「現在のフレームに教師を重ねる」で枠と教師の認識を確認してから生成する。足上げ・前方への蹴り出し・足裏・交差・部分遮蔽の映像を含める。教師が間違った場面まで正解にできる機能ではない。
同じ動画は学習用と評価用へ重複登録できない。評価には別撮影を使う。学習は左右反転・明るさ・一部遮蔽を加える。反転時は左右関節・X分布・正解ラベルも対応させる。映像はサーバーへ送信しない。
## 学習・再開
「蒸留を開始・続行」を押す。教師セッションは学習前に解放し、学生のためにGPUメモリを空ける。学習率は100更新で指定値まで増やす。各更新で勾配ノルムをGPU上で計算し、損失・ノルム診断を一括読戻ししてからクリッピングした勾配を適用する。重みごとのCPUコピーは行わない。
60秒ごとに重み・Adam一次/二次モーメント・累積係数・乱数・シャッフル順・カーソル・更新数・学習時間をIndexedDBへ保存する。学習30分ごとに同じ情報の復元JSONを自動ダウンロードする。ブラウザの複数ダウンロード許可が必要な場合がある。学習中のページ離脱には確認を出す。
復元JSONを読み込むとモデルを準備し、同じ教師キャッシュから自動再開する。手入力したバッチ・学習率・目標回数はJSONより優先する。JSONに画像や数GBの教師分布は含めないため、別PCへの移行時は同じ教師・画像・動画から教師キャッシュを再生成する必要がある。WB73の重みは別アーキテクチャなのでこの新モデルへロードしない。旧ページで引き続き使用できる。
## 推論・ONNX
「画像または現在の動画フレームで推論」で学生の出力を切り抜き画像へ重ね、ウォームアップ後のWebGPU時間を表示する。停止姿勢だけではなく、足上げ・蹴り出しのフレームで確認する。
評価画面の教師XY/Z差は教師への一致度であり、実測3D正解への誤差ではない。正解ラベルがあるCOCO/CMU画像では別に2D PCK@0.05を表示する。
「軽量ONNXを保存」で `distill37.onnx` と出力仕様JSONを得る。入力 `image`: float32 `[1,3,256,192]` RGB 0〜1。出力 `keypoints`: `[1,37,3]`、XYは192×256切り抜き内ピクセル、Zは骨盤基準メートル。`scores`: `[1,37]` はXY分布の集中度。
XYピクセルとZメートルをそのまま同じ空間のXYZとしてOSCへ送ってはいけない。カメラ・身体スケール・VR空間への変換は推論アプリ側で行う。この変更は蒸留モデルとその学習・出力までを実装し、送信ドライバーは追加しない。
## 確認範囲
Node.js上のTensorFlow.js CPUで順伝播・逆伝播・Adam更新を実行した。足上げ実映像1フレームからRTMW3D-Xの生SimCCを取り出し、37点の教師分布を生成し、学生の1更新で損失4.40696→4.26823となった。ONNX Runtime Nodeとの出力比較は座標差0、スコア差の最大0.00000191。推論はOS側でネットワーク通信を禁止して実行した。この短い実行を認識精度の達成として扱わない。
WebGPUのブラウザ実行、長時間の実データ学習後の精度、RTX 4070 Ti SUPERでの速度は未確認。既存のSiteにはプレビュー用dev設定がないため、ブラウザ実行済みとはしていない。公式.pthの取得・ハッシュ確認は実施済み。Python 3.10の新規変換スクリプトはこの環境では未実行で、実際の教師分布確認には既存の公式重み由来FP32 ONNXを使用した。
## ライセンス・出典
MMPose / RTMW3D: Apache-2.0。公式モデル定義と重みのURL・revision・ハッシュは変換JSONへ保存する。TensorFlow.js: Apache-2.0。ONNX Runtime: MIT。既存画像・動画はそれぞれの利用条件を引き継ぐ。
