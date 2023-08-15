from .collect import job_simple, job_specific


def collector(args):
    js = job_simple(args)
    if js.job_simple_path.is_file() and js.job_specific_path.is_file():
        print('raw data already exists')
        return 
    else:
        response = input('raw data does not exist. Do you want to collect raw data? (y/n)')
        if response == 'y':
            df1 = js.parse_jobs_L()
            df2 = job_specific(args, df1['구인인증번호']).parse_jobs_D()
            print('data collection is done')
        else:
            print('exit')
            exit()