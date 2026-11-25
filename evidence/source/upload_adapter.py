EXPECTED_COLUMNS={'State_Name','District_Name','Crop_Year','Season','Crop','Area','Production'}
def configure_dataset(csv_path=None,upload_callback=None):
    if upload_callback is not None:
        uploaded=upload_callback()
        names=[name for name in uploaded if name.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError('Upload exactly one crop_production.csv file.')
        path=Path(names[0]).name
        Path(path).write_bytes(uploaded[names[0]])
    else:
        path=str(csv_path or os.environ.get('APY_CSV','crop_production.csv'))
    if not Path(path).is_file():raise FileNotFoundError('Set APY_CSV or upload crop_production.csv.')
    import csv
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:header=next(csv.reader(handle))
    if set(header)!=EXPECTED_COLUMNS:raise ValueError(f'Expected seven-column crop_production.csv; received {header}')
    os.environ['APY_CSV']=str(Path(path).resolve())
    return Path(path)
if COLAB_MODE:
    from google.colab import files
    data_path=configure_dataset(upload_callback=files.upload)
    os.environ['APY_OUTPUT']='/content/crop_yield_final_results'
else:
    data_path=configure_dataset()
    os.environ.setdefault('APY_OUTPUT','results')
print('Dataset:',data_path.name)
