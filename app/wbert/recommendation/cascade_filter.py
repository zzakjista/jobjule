from haversine import haversine 
import pandas as pd 
import numpy as np 

class cascade_filtering:

    def __init__(self, args, recommendation_dictionary, export_path):

        self.recommendation_dictionary = recommendation_dictionary
        self.export_path = export_path
        self.job_simple_path, self.job_specific_path = self.export_path._get_rawdata_datasets_path()
        self.geo_site_path = self.export_path._get_geosite_datasets_path()
        self.job_simple = pd.read_csv(self.job_simple_path)
        self.job_specific = pd.read_csv(self.job_specific_path)
        self.geo_site = pd.read_csv(self.geo_site_path)
        self._ppc_geo_site()

        self.job_data = self.job_simple.merge(self.job_specific, how='left', on='구인인증번호')
        self.geo_data = self.job_simple.merge(self.geo_site, how='left', left_on='근무지역', right_on='행정구역')
        self.geo_data.drop(['행정구역'], axis=1, inplace=True)
        print('getting distance matrix...')
        self.distance_matrix = self._get_distance_matrix()
        print('complete distance matrix')
        self.sudo_list = ['서울', '경기', '인천']
        self.sudo_dist = args.sudo_dist
        self.province_dist = args.province_dist

    def filtering(self):
        self.filtering_region()
        self.filtering_edu()
        self.filtering_career()
        return self.recommendation_dictionary

    def filtering_career(self):
        job_data = self.job_data.copy()
        job_data['경력'].fillna(job_data['경력'].mode()[0], inplace=True)
        job_data  = job_data[['구인인증번호', '경력']]
        job_data.set_index('구인인증번호', inplace=True)
        career_dict = job_data['경력'].to_dict()
        for key, values_list in self.recommendation_dictionary.items():
            if career_dict[key] == '신입':
                filtered_values = [val for val in values_list if (career_dict[val] != '경력')]
            else:
                filtered_values = values_list
    
            self.recommendation_dictionary[key] = filtered_values

    def filtering_edu(self):
        job_data = self.job_data.copy()
        job_data.set_index('구인인증번호', inplace=True)
        job_data['학력'].fillna(job_data['학력'].mode()[0], inplace=True)
        job_data['최소학력'] = job_data['학력'].str.split('-').str[0]

        edu_mapping = {
        '학력무관' : 0, '초졸이하' : 1, '중졸' : 2, '고졸' : 3, 
        '대졸(2~3년)' : 4, '대졸(4년)' : 5, '석사' : 6, '박사' : 7
        }

        job_data['edu_mapped'] = job_data['최소학력'].map(edu_mapping)
        edu_dict_mapped = job_data['edu_mapped'].to_dict()
        for key, values_list in self.recommendation_dictionary.items():
            key_edu = edu_dict_mapped[key]
            if key_edu == 0:
                filtered_values = values_list
            else:
                filtered_values = [val for val in values_list if key_edu >= edu_dict_mapped[val]]
            self.recommendation_dictionary[key] = filtered_values

    def filtering_region(self):
        df = self.geo_data[['구인인증번호','근무지역','위도','경도']]
        for key, values_list in self.recommendation_dictionary.items():
            key_region = df.loc[df['구인인증번호'] == key, '근무지역'].values[0]

            key_dist = self.distance_matrix[key_region]
            filter_mask = (key_dist <= self.sudo_dist) if key_region[:2] in self.sudo_list else (key_dist <= self.province_dist)
            filter_mask = df['구인인증번호'].isin(values_list) & df['근무지역'].isin(list(filter_mask[filter_mask==1].index))
            filtered_value = df.loc[filter_mask, '구인인증번호'].tolist()
            self.recommendation_dictionary[key] = filtered_value

    def _get_distance_matrix(self):
        geo_unique = self.geo_data.drop_duplicates(subset=['근무지역'], ignore_index = True)
        geo_unique = geo_unique.replace('', 0) # 결측치를 0으로 채우면 필요없음
        geo_unique = geo_unique.astype({'위도' : 'float32'}, {'경도' : 'float32'})
        # mat = np.zeros((geo_unique.shape[0], geo_unique.shape[0]))
        lst = []
        print('list is created')
        coords = geo_unique[['위도', '경도']].values
        for i in range(geo_unique.shape[0]):
            for j in range(geo_unique.shape[0]):
                lst.append(haversine(coords[i], coords[j]))

        mat = np.array(lst).reshape(geo_unique.shape[0], geo_unique.shape[0])
                # mat[i,j] = haversine(coords[i], coords[j])
        distance_matrix = pd.DataFrame(mat, index = geo_unique['근무지역'], columns = geo_unique['근무지역'])
        return distance_matrix

    def _ppc_geo_site(self):
        self.geo_site.fillna('', inplace = True)
        geo_dict = {'경상남도':'경남','경상북도':'경북','전라남도':'전남','전라북도':'전북','충청남도':'충남','충청북도':'충북'}
        for i in geo_dict.keys():
            self.geo_site['시도'] = self.geo_site['시도'].str.replace(i,geo_dict[i])
            
        self.geo_site['시도'] = self.geo_site['시도'].str[:2]
        self.geo_site['행정구역'] = (self.geo_site['시도'] + ' ' + self.geo_site['시군구'] + ' ' + self.geo_site['읍면동/구'] + ' ' + self.geo_site['읍/면/리/동'] + ' ' + self.geo_site['리']).str.strip()
        self.geo_site = self.geo_site[['행정구역','위도','경도']]




        