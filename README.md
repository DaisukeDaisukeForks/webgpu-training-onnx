# WholeBody73 Local Inference Lab
指定された `c513f6e24e0ef0d10a98545086d4a90968474074` から分岐した推論専用サイト。学習サイトとは別のSitesプロジェクトで動作する。
## 使い方
1. 学習チェックポイントのJSONまたはgzip圧縮JSONを選択する。ファイル名の制約はない。内容のformat/schemaと全重みの名前・形状・有限値を検証する。
2. ブラウザでONNXを生成し、ONNX Runtime Webの推論セッションを作成する。「生成したONNXを保存」で取得できる。
3. 動画を選択する。VRMは任意。「現在フレームを推論」または「推論開始」を選ぶ。
4. 複数人物の検出枠、73関節、3D骨格、腰基準のZ範囲と概算配置距離を確認する。VRMは人物ごとに読み込み、腕・脚・腰を駆動する。
## 入出力と制約
- ファイルはすべて端末内で読み込む。動画・VRM・チェックポイント・生成ONNXをサーバーへ送信しない。チェックポイントはリポジトリに含めない。
- 初回にCDNからThree.js、three-vrm、TensorFlow.js、COCO-SSD、ONNX Runtime Webおよび人物検出重みを取得する。VRMの外部URIリソース参照は拒否する。
- 学習側のmodel.mjs / onnx.mjs / video-model.mjs / schema.mjsを使用する。ランダム重みで補完せず、完全なチェックポイントだけを変換する。動画モデルではIDごとの3フレーム履歴を使う。
- 複数人処理はCOCO-SSDによる検出とIoUによる簡易ID割り当ての後、各人物を1.25倍・縦横比0.75で切り抜き、同一のONNXモデルで順次推論する。交差・長い遮蔽でIDは入れ替わりうる。
- 出力座標は学習モデルのUnity軸（+X右、+Y上、+Z前方、腰中心、m）。表示時にZを反転してThree.jsの右手座標へ変換する。
- 関節の奥行きはモデル出力。人物全体の配置距離は `focal_px × assumed_height / bbox_height_px` による概算であり、絶対深度の測定ではない。水平画角と推定身長を調整できる。
- confidenceは学習側定義のヒートマップ集中度であり、校正された可視性確率ではない。品質の保証や姿勢の補完は行わない。初期表示しきい値0は未学習の出力も見えるようにするため。
- GPU非対応・セッション作成失敗時はWASMへフォールバックする。端末速度・人数・動画によってリアルタイム速度に届かない場合がある。
- 埋め込みVRM 0.x/1.0の主要ボーンのみを駆動し、指・表情のリターゲットは実装対象外。
## 配布とライセンス
このリポジトリのコードはLICENSEに従う。添付学習コード由来の4モジュールは元プロジェクトの著作権表示を維持する。
- Three.js: MIT — https://github.com/mrdoob/three
- three-vrm: MIT — https://github.com/pixiv/three-vrm
- TensorFlow.js / COCO-SSD: Apache-2.0 — https://github.com/tensorflow/tfjs / https://github.com/tensorflow/tfjs-models
- ONNX Runtime: MIT — https://github.com/microsoft/onnxruntime
入力チェックポイントの学習データおよび入力VRM・動画に適用される利用条件は、各権利者の条件に従う。
