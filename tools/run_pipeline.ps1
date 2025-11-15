param(
  [string]$ConfigPath = "configs/params.json",
  [string[]]$Steps = @("data","train","distill","quantize","prune","onnx")
)

$ErrorActionPreference = "Stop"

function Read-Config($path) {
  if (-not (Test-Path $path)) { throw "Config not found: $path" }
  $json = Get-Content $path -Raw | ConvertFrom-Json
  return $json
}

function Activate-Venv() {
  $activate = Join-Path (Get-Location) "venv/Scripts/Activate.ps1"
  if (Test-Path $activate) { & powershell -ExecutionPolicy Bypass -File $activate }
}

function Run($cmd) {
  Write-Host "==> $cmd"
  & powershell -ExecutionPolicy Bypass -Command $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

$cfg = Read-Config $ConfigPath
Activate-Venv

if ($Steps -contains "data") {
  Run "python -m src.data.generate_goals --count $($cfg.data.count) --noise_rate $($cfg.data.noise_rate) --emoji_rate $($cfg.data.emoji_rate) --freeform_rate $($cfg.data.freeform_rate) --output_dir $($cfg.data.output_dir)"
}

if ($Steps -contains "train") {
  Run "python -m src.training.finetune_bert --data_dir $($cfg.train.data_dir) --epochs $($cfg.train.epochs) --batch_size $($cfg.train.batch_size) --max_length $($cfg.train.max_length) --output_dir $($cfg.train.output_dir) --limit_train $($cfg.train.limit_train) --limit_eval $($cfg.train.limit_eval) --grad_accum $($cfg.train.grad_accum)"
}

if ($Steps -contains "distill") {
  Run "python -m src.training.distill_student --data_dir $($cfg.distill.data_dir) --teacher_name $($cfg.distill.teacher_name) --student_name $($cfg.distill.student_name) --epochs $($cfg.distill.epochs) --batch_size $($cfg.distill.batch_size) --temperature $($cfg.distill.temperature) --alpha $($cfg.distill.alpha) --output_dir $($cfg.distill.output_dir) --limit_train $($cfg.distill.limit_train) --limit_eval $($cfg.distill.limit_eval) --grad_accum $($cfg.distill.grad_accum)"
}

if ($Steps -contains "quantize") {
  Run "python -m src.optimization.quantize --model_dir $($cfg.quant.model_dir) --data_dir $($cfg.quant.data_dir) --output_dir $($cfg.quant.output_dir) --max_length $($cfg.quant.max_length)"
}

if ($Steps -contains "prune") {
  Run "python -m src.optimization.prune --model_dir $($cfg.prune.model_dir) --data_dir $($cfg.prune.data_dir) --output_dir $($cfg.prune.output_dir) --amount $($cfg.prune.amount) --max_length $($cfg.prune.max_length)"
}

if ($Steps -contains "onnx") {
  Run "python -m src.export.export_onnx --model_dir $($cfg.onnx.model_dir) --data_dir $($cfg.onnx.data_dir) --output_dir $($cfg.onnx.output_dir) --opset $($cfg.onnx.opset) --max_length $($cfg.onnx.max_length) --samples $($cfg.onnx.samples)"
}

Write-Host "Pipeline completed."