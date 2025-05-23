from primary_plots import get_result_db_details_by_rd, filter_none_nan



test_comb, val_comb, time_param_details = get_result_db_details_by_rd(
    reduce_data=0.8,
    get_time_param_details=True,
    task='custom')


val_filtered = filter_none_nan(val_comb)
test_filtered = filter_none_nan(test_comb)
test_filtered = test_filtered[test_filtered['train_size'] <= 3000]
print(f'Not None dataset found: {test_filtered.shape[0]}\n')

print('-'*30)
print('Result for custom datasets')
print(test_filtered.iloc[:,:-4])
