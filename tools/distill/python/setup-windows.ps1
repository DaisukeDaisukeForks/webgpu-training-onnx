$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../../..')
function Run([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
Run 'py' @('-3.10', '-m', 'venv', '.venv-prune')
$Python = Join-Path (Get-Location) '.venv-prune/Scripts/python.exe'
Run $Python @('-m', 'pip', 'install', 'pip==24.0', 'setuptools==69.5.1', 'wheel')
Run $Python @('-m', 'pip', 'install', 'torch==2.1.0', 'torchvision==0.16.0', '--index-url', 'https://download.pytorch.org/whl/cu118')
Run $Python @('-m', 'pip', 'install', '--no-build-isolation', '-r', 'tools/distill/python/requirements.txt')
Run $Python @('-m', 'pip', 'install', '--no-deps', '--only-binary=:all:', 'mmcv==2.1.0', '-f', 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html')
$Revision = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
if (!(Test-Path 'teacher-upstream')) {
    Run 'git' @('clone', 'https://github.com/open-mmlab/mmpose.git', 'teacher-upstream')
    Run 'git' @('-C', 'teacher-upstream', 'checkout', $Revision)
}
$Actual = & git -C teacher-upstream rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $Actual.Trim() -ne $Revision) { throw 'teacher-upstream must contain the pinned official revision; existing files were left untouched.' }
Run $Python @('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', 'teacher-upstream')
Run $Python @('-c', 'import torch,mmcv,numpy; print("torch",torch.__version__,"CUDA",torch.version.cuda,"cuDNN",torch.backends.cudnn.version(),"GPU available",torch.cuda.is_available())')
