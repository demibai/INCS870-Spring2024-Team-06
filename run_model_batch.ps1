$methods = "anova", "chi2"
$startK = 10
$endK = 10
$step = 1

foreach ($method in $methods) {
    for ($k = $startK; $k -le $endK; $k += $step) {
        python train_eval.py method=$method k=$k
    }
}

Write-Output "Done running model for k=$startK to k=$endK with step=$step using method=$method"