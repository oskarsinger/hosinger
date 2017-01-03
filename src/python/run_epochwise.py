import click

from runners.wavelets import EpochWiseTimeSeriesAnalysis as EWTSA
from runners.wavelets import ViewPairwiseCorrelationRunner as VPWCR
from runners.wavelets import MVDTCWTRunner

@click.command()
@click.option('--data-path')
@click.option('--save-load-dir')
@click.option('--dataset', default='e4')
@click.option('--wavelet-dir')
@click.option('--load', default=False)
@click.option('--save', default=False)
@click.option('--show', default=False)
def run_it_all_day_bb(
    data_path,
    save_load_dir,
    dataset,
    wavelet_dir,
    load,
    save,
    show,):

    dtcwt_runner = MVDTCWTRunner(
        data_path=data_path,
        dataset=dataset,
        save_load_dir=wavelet_dir,
        load=True)

    dtcwt_runner.run()

    def get_analysis_runner(
	epoch,
	dtcwt_runner,
	save_load_dir,
	save,
	load,
	show):
	
	return VPWCR(
	    dtcwt_runner,
	    save_load_dir,
	    wavelets=epoch,
	    save=save,
	    load=load,
	    show=show)

    boundaries = [2,5]

    runner = EWTSA(
	dtcwt_runner,
	save_load_dir,
	get_analysis_runner,
	boundaries,
	save=save,
	load=load,
	show=show)	

    runner.run()

if __name__=='__main__':
    run_it_all_day_bb()
