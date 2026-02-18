param(
    [string]$Config = "config/dataset_config.json",
    [string]$Output = "data/synth_surface_defects",
    [int]$NumImages = 1000,
    [int]$Seed = 42
)

python -m src.pipeline.generate_dataset --config $Config --output $Output --num-images $NumImages --seed $Seed
