import psycopg2 as pg2
import time

file_list= ['clean_2000.csv','clean_2001.csv','clean_2002.csv',
            'clean_2003.csv','clean_2004.csv','clean_2005.csv',
            'clean_2006.csv','clean_2007.csv','clean_2008.csv']
name_list = ['w_00','w_01','w_02','w_03','w_04','w_05','w_06','w_07','w_08',]
directory_path='/Users/Berzyy/plant_forecast/data/weather/clean_csv/'
db_name = 'weather'

def upload_csv_to_postgres(file_list, name_list, directory_path, db_name, loc='localhost'):
    """INPUTS
        file_list = list of files a list of strings i.e. ["2001_clean.csv","2002_clean.csv"]
        name_list = list of names you want to call the databases ['2001_weather','2002_weath']
        directory_path = The path to the folder the files are stored in (absolute)
        '/Users/Berzyy/plant_forecast/data/weather/'
        db_name = the name of data base you want to connect to
        host = host
    """
    conn = pg2.connect(dbname=db_name, host=loc)
    cur = conn.cursor()
    conn.autocommit=True
    for name, file in zip(name_list,file_list):
        start = time.time()
        path_to_file= directory_path+file

        make_table_command = f"""CREATE TABLE {name}
                        (index int,
                         station_id text,
                         measurement_date date,
                         measurement_type text,
                         measurement_flag int); """

        upload_data_command= f"""COPY {name}
                                FROM '{path_to_file}'
                                csv header;"""

        cur.execute(make_table_command)
        cur.execute(upload_data_command)

        run_time= time.time()-start
        print(f'Uploaded Weather Data: {run_time}')
    conn.close()
    return None
