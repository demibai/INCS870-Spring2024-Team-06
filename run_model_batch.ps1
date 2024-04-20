$task = "multi"
# $methods = "chi2", "anova", "mutual_information", "rfe"
$methods = "variance_threshold"
$startK = 1000
$endK = 1000
$step = 10
$start_time = Get-Date

foreach ($method in $methods) {
    for ($k = $startK; $k -le $endK; $k += $step) {
        python train_eval.py method=$method k=$k task=$task
    }
}

$end_time = Get-Date
$elapsed_time = $end_time - $start_time

Write-Output "Done running model for k=$startK to k=$endK with step=$step using methods=$methods"
Write-Output "Elapsed time: $elapsed_time"