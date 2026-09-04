# ComfyUI H3 Prompt Optimizer

MiniMax H3 Ref2VA の生成動画を人間がレビューし、局所的な Prompt IR Patch を承認して再生成するためのカスタムノードです。既存ノードは変更せず、次の4ノードとサイドバーを追加します。

- `H3 Optimizer Segment Settings`（共通設定）
- `H3 Optimizer Ref2VA Prompt Package Generator`（推奨）
- `H3 Optimizer Ref2VA Prompt Package`
- `H3 Optimizer Video Output`
- `H3 Prompt Optimizer` サイドバー

## 接続

新規ワークフローでは、GeneratorとOptimizerを統合した`H3 Optimizer Ref2VA Prompt Package Generator`を使用します。参照素材、`reference_notes`、`instruction`はこのノードだけに入力します。

```text
H3 Optimizer Segment Settings.optimizer_id
  ├─> H3 Optimizer Ref2VA Prompt Package Generator.optimizer_id
  └─> H3 Optimizer Video Output.optimizer_id

H3 Optimizer Segment Settings.project_name
  ├─> Video Segment Checkpoint.project_name
  └─> H3 Optimizer Video Output.project_name

H3 Optimizer Segment Settings.segment_id
  ├─> Video Segment Checkpoint.segment_id
  └─> H3 Optimizer Video Output.segment_id

H3 Optimizer Segment Settings.resume_from
  ├─> Video Segment Checkpoint.resume_from
  └─> H3 Optimizer Video Output.resume_from

H3 Optimizer Segment Settings.continuation_frame_count
  ├─> Video Segment Checkpoint.continuation_frame_count
  └─> Video Segment Checkpoint.trim_frames（Segment 2以降）

H3 Optimizer Segment Settings.blend_frames
  └─> Video Segment Checkpoint.blend_frames

参照素材 ──> H3 Optimizer Ref2VA Prompt Package Generator.ref_*

H3 Optimizer Ref2VA Prompt Package Generator.prompt
  ──> Persistent Text.text
  ──> MiniMax H3 Reference to Video from Package.prompt

H3 Optimizer Ref2VA Prompt Package Generator.package
  ──> MiniMax H3 Reference to Video from Package.package
```

Settingsの`optimizer_id`は`<project_name>_segment_<3桁以上のsegment_id>`として自動生成されます。例えば`project_name=my_movie`、`segment_id=1`なら`my_movie_segment_001`です。接続先がwidget表示の場合は、右クリックして`Convert Widget to Input`してから接続します。

`use_approved_prompt=false`では、既存の`MiniMax H3 Ref2VA Prompt Package Generator`と同じ処理を内部で実行して初回Promptを生成します。`use_approved_prompt=true`ではPrompt生成処理とそのLLM通信を実行せず、同じ`ref_*`入力と`approved_prompt`からPackageをローカルで再構築します。Sidebarの`Apply`と`Apply & Generate`は、この2つのwidgetを自動更新します。

参照素材の接続数と承認Prompt内の`<Picture n>`、`<Video n>`、`<Audio n>`が一致しない場合は、参照欠落として停止します。統合ノードは新PromptとPackage内Promptを同時に再バインドするため、既存Ref2VA consumerの厳密なPrompt/Package一致検査を維持できます。`Persistent Text`は統合ノードの後段に置きます。

既存ワークフロー向けの`H3 Optimizer Ref2VA Prompt Package`もそのまま利用できます。この分離構成では、既存Generatorの`prompt`と`package`に加え、承認モード用の同じ参照素材を`approved_ref_*`へ接続します。新規ワークフローでは、参照の二重配線が不要な統合ノードを推奨します。

生成動画の記録は次のように接続します。

```text
Video Segment Checkpoint.video
  ──> H3 Optimizer Video Output.video

H3 Optimizer Ref2VA Prompt Package Generator.prompt_state
  ──> H3 Optimizer Video Output.prompt_state
```

`project_name`、`segment_id`、`resume_from`、`continuation_frame_count`、`blend_frames`はSettingsを設定元とします。複数segmentではSegmentごとにSettingsノードを1つ配置してください。先頭Segmentは通常 `continuation_frame_count=22 / blend_frames=0`、後続Segmentは `22 / 5` とし、後続Segmentだけ `continuation_frame_count` をCheckpointの `trim_frames` にも接続します。Sidebarの`Apply & Generate`は、接続されたSettingsの`resume_from`を更新します。

Checkpoint が LOAD する segment では Video Output は `prompt_state` を評価しません。GENERATE された segment だけを Registry に登録します。Checkpoint が過去 revision を整理してもレビュー履歴が消えないよう、動画は `output/h3_prompt_optimizer/artifacts/` に独立保存されます。

`H3 Optimizer Video Output`は、登録した動画ごとに次の再生成情報も同じartifactディレクトリへ保存します。

- 実際にサーバーへ送られたAPI prompt（sampler seed、Prompt Generator seed、steps、解像度などを含む）
- Queue時点のComfyUI workflow

`Video Segment Checkpoint`は現在有効なSegment動画と境界フレームを保持し、途中Segmentからの再開を担当します。生成条件の履歴と復元は`H3 Optimizer Video Output`が担当します。

## VLM 設定

解析には、画像入力対応の OpenAI-compatible `/v1/chat/completions` またはローカルの`llama-cli`を使用できます。既定はOpenAI-compatible方式で、URLの既定値は`http://localhost:1234`です。VLM設定にはPrompt Generatorと共通の`MINIMAX_*`キーを使用します。Optimizerでの設定の優先順位は次の通りです。

1. プロセス環境変数の`H3_OPTIMIZER_*`、次に`MINIMAX_*`
2. このディレクトリの`.env`にある`H3_OPTIMIZER_*`、次に`MINIMAX_*`
3. `ComfyUI-MiniMaxH3-Prompt-Generator/.env`にある`MINIMAX_*`
4. 既定値

同じ設定を両プラグインで使用する場合は、Prompt Generator側の`.env`だけに`MINIMAX_*`を設定し、Optimizer側では該当キーを省略できます。両ディレクトリの`.env`に同じキーがある場合、ファイルは上書きされません。Optimizerは自身の`.env`の値を、Prompt Generatorは自身の`.env`の値を使用するため、意図的に別設定にすることもできます。

既存の`H3_OPTIMIZER_*`キーも後方互換性のため利用でき、対応する`MINIMAX_*`より優先されます。OptimizerだけバックエンドやCLIモデルを変更したい場合の上書きに使用してください。`H3_OPTIMIZER_MODEL`、`H3_OPTIMIZER_API_KEY`、`H3_OPTIMIZER_TIMEOUT_SECONDS`はCritic/PlannerのHTTP要求に固有の設定です。`H3_OPTIMIZER_MODEL`を省略するとサーバー側のロード済みモデルを使用します。

```text
MINIMAX_BASE_URL=http://localhost:1234
H3_OPTIMIZER_MODEL=Qwen3-VL-8B-Instruct
H3_OPTIMIZER_TIMEOUT_SECONDS=300
```

ローカルの`llama-cli`を使用する場合は、`.env`へ次の設定を追加します。モデルと`mmproj`には対応する画像入力対応GGUFを指定してください。

```text
MINIMAX_VLM_BACKEND=llama_cli
MINIMAX_LLAMA_CLI_PATH=/absolute/path/to/llama-cli
MINIMAX_LLAMA_MODEL_PATH=/absolute/path/to/vlm-model.gguf
MINIMAX_LLAMA_MMPROJ_PATH=/absolute/path/to/mmproj-model.gguf
```

CLI方式では解析開始前にComfyUIのモデルをVRAMからアンロードします。ComfyUIの実行中・待機中Queueがある場合と、別のローカル解析が実行中の場合は開始しません。各LLM要求で`llama-cli`を1回起動し、終了時にモデルを解放します。通常の解析ではCriticとPlannerなどで複数回起動することがあります。ローカル解析中は新しいComfyUI Queueを開始しないでください。次回の通常Queueでは必要なComfyUIモデルが自動的に再ロードされます。

Prompt Generator側の`.env`に共通設定を置けば、サイドバーのCritic/Planner解析と`use_approved_prompt=false`で実行される初回Prompt生成の両方に適用されます。両プラグインのローカルCLI推論は共有ロックにより同時実行されません。

CLI方式では、チャットテンプレートの生成prefixと`llama.cpp`のJSON Schema文法が衝突しないよう、汎用JSONオブジェクト文法で生成を制約した後にPydanticで要求スキーマを厳密に検証します。VLM応答がJSONまたは要求スキーマに一致しない場合は、初回生成と修正生成を区別し、スキーマ名、応答の先頭1600文字、`llama-cli`診断出力の末尾1600文字をComfyUIログへ記録します。応答が空の場合は、画像数、生成トークン上限、コンテキストサイズとCLIのタイミング情報も記録します。

## 操作

1. 通常どおり Queue して動画を生成する。
2. `H3 Prompt Optimizer` サイドバーで `optimizer_id` と生成履歴を選び、動画、Prompt、seedなどの生成条件を確認する。
3. 完全に同じ条件をCanvasへ戻す場合は`生成条件を復元`、そのままQueueする場合は`同じ条件で再生成`を押す。
4. 低品質プレビューから高品質版を作る場合は`生成条件を復元`後、解像度とsampler stepsだけを変更してQueueする。
5. Promptを改善する場合は、改善点と必要なら開始秒・終了秒を入力して`動画を解析`を押す。
6. 観測結果、提案 action、Prompt 差分を確認し、`Apply`または`Apply & Generate`を押す。

同じPrompt・seedでも、解像度やstepsを変更すると拡散計算そのものが変わるため、低品質版とフレーム単位で同一の動画にはなりません。完全な再現には、保存した条件に加えてモデルファイル、参照素材、ComfyUI/custom node、計算backendも同じである必要があります。

`PATCH_PROMPT` と `REGENERATE_SAME_PROMPT` だけが自動 Apply できます。Criticの原因分類は参考情報としてPlannerが再評価するため、`MODEL_CAPABILITY_LIMIT`と判定されても、Promptの不足・矛盾を許可範囲で修正できる場合はPatchを提案します。参照変更、証拠不足、Promptでも再生成でも対処できない問題は自動変更しません。解析後に Prompt ノードやその上流が変更された場合や、新しい動画が登録された場合も、解析結果のPromptを現在の対象ノードへApplyします。

## 解析と保存

- 時間範囲あり: 前後0.5秒を含む最大24 FPSの focused scan
- 時間範囲なし: 最大4 FPSの global scan。時間問題を検出した場合だけ最大24 FPSで再解析
- 最大48フレーム、長辺768 px JPEGをVLMへ送信
- Prompt 全文の自由書き換えは禁止し、対象 Shot body または音声 section の許可パスだけをPatch
- `appearance`、`identity_consistency`、`object_consistency`では、内部矛盾を残さないため`subject_definitions`と`retention_analysis`もPatch可能。参照ラベルの追加・削除は禁止
- Prompt、Prompt IR、Critic結果、Patch、差分、世代関係を SQLite に保存
- 動画、実行済みAPI prompt、workflowを世代別artifactとして保存

現在の PoC は映像フレームを Critic へ渡します。音声波形そのもの、Optical Flow、Pose、Tracking はまだ解析しません。

## テスト

ComfyUI の仮想環境で次を実行します。追加パッケージは不要です。

```powershell
.\.venv\Scripts\python.exe custom_nodes\ComfyUI-H3-Prompt-Optimizer\tests\test_optimizer.py
```
