def cast_array_to_csv(timestamp_array, ndvi_array, product, tile):
    """INPUT:
    Timestamp is a 2d array,
    NDVI is a 3d array
    """
    df = pd.DataFrame()
    df['capture_date']= timestamp_array
    df['product']= product
    df['tile_id']= tile
    df['ndvi']= ndvi_array

    df['capture_date'] = pd.to_datetime(df['capture_date'])
    df['product']=df['product'].astype(str)
    df['tile_id']=df['tile_id'].astype(str)
    df['ndvi']=df['ndvi']
    return df

def JulianDate_to_MMDDYYY(y,jd):
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    return datetime.date(y, month, jd)

def quality_screen(quality, ndvi):
    """INPUTS
        quality= 2D array
        ndvi = 2d array
    """
    ndvi[quality!=0]= -3000
    return ndvi


def pixel2coord(x, y):
    """Returns global coordinates from pixel x, y coords"""
    xoff= -117.4740487
    yoff= 39.9958333
    a= 0.008868148103055
    b= 0
    d= 0
    e=-0.008868148103054807

    lat = a * x + b * y + xoff
    long = d * x + e * y + yoff

    return lat, long

def try_again(path):
    """Takes in a path to a folder where there are two folders:
        1.) ndvi_tiff
        2.) quality_tiff
    Takes in this folder and casts NDVI values and Quality Values into 2d arrays
    """
    quality_folder_path= path+ 'quality_tiff/'
    ndvi_folder_path= path+ 'ndvi_tiff/'
    file_list= os.listdir(ndvi_folder_path)

    ndvi_file_set=  set(list(f for f in os.listdir(ndvi_folder_path) if f.endswith('.' + 'tif')))
    quality_file_set= set(list(f for f in os.listdir(quality_folder_path) if f.endswith('.' + 'tif')))

    latitude, longitude = make_coordinate_array()
    maybe = sorted(ndvi_file_set&quality_file_set)
    for f in maybe:
        start= time.time()
        product= f[:7]
        year= f[9:13]
        julian_day= f[13:16]
        tile= f[17:23]

        ndvi_file= ndvi_folder_path+f
        quality_file= quality_folder_path+f

        ndvi = gdal.Open(ndvi_file)
        n_band = ndvi.GetRasterBand(1)
        n_arr = n_band.ReadAsArray()
        ndvi= None

        quality = gdal.Open(quality_file)
        q_band = quality.GetRasterBand(1)
        q_arr = q_band.ReadAsArray()
        quality= None

        data=quality_screen(q_arr, n_arr)
        #return data
        av=data[data!=-3000].mean()
        date_time=JulianDate_to_MMDDYYY(int(year),int(julian_day))

        date = np.full(data.shape, date_time)

        tile_matrix = np.matrix([date[0][0], -104.9256192, 34.9986319, av])
        #matrix_list.append(tile_matrix)
        print(f'Compiled Matrix for {date_time} in {time.time()-start} seconds')
        #return matrix_list
        #yield(tile_matrix)
        return tile_matrix



def matrix_compiler(folder_path):
    it= try_again(folder_path)
    df=pd.DataFrame(None)
    final= np.array([]).reshape(0,4)
    for matrix in it:
        start=time.time()

        print(matrix.shape)
        print(final.shape)
        final = np.vstack((final,matrix))
        print(time.time()-start)
    return final

def construct_matrix(time_list, ndvi_array, lat_list, long_list):
    """ Takes in time, ndvi, lat_long, and a value for NDVI
        creates a matrix with
        DATE | LAT | LONG | Value
    """
    time_array= np.array(time_list)
    df = pd.DataFrame()
    index=0
    matrix_list= []
    for tile in ndvi_array:

        flat = tile.flatten().reshape(1,-1)
        latitude= np.array(lat_list).reshape(flat.shape)
        longitude= np.array(long_list).reshape(flat.shape)
        time = np.full(flat.shape, time_list[index])
        index +=1
        tile_matrix= np.transpose([time, latitude, longitude, flat])

        matrix_list.append(tile_matrix)
    return matrix_list

def cast_csv_to_postgres(path_to_file, db_name, table_name, loc='localhost'):
    """Takes in a data frame with columns:
    capture_date = date time object
    product = i.e. MOD13A2
    tile_id = i.e. h09v05
    ndvi = 2d numpy array
    """
    conn = pg2.connect(dbname=db_name, host=loc)
    cur = conn.cursor()
    conn.autocommit=True
    make_table_command = f"""CREATE TABLE {table_name}
                                (index int,
                                capture_date date,
                                product text,
                                tile_id text,
                                ndvi integer[][]);"""



    upload_data_command= f"""COPY {table_name}
                                FROM '{path_to_file}'
                                csv header;"""

    cur.execute(make_table_command)
    cur.execute(upload_data_command)

    #run_time= time.time()-start
    #print(f'Uploaded MODIS Data: {f to SQL in about {run_time} seconds')
    conn.close()
    return None
            
