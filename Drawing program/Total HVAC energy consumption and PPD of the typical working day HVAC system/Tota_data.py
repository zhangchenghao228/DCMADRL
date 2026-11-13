import pandas as pd
import os
def read_and_combine(excel_sources, current_directory, prefix=True):
    """
    从多个 Excel 文件中读取指定的工作表和列，并合并成一个 DataFrame。
    
    参数：
    - excel_sources: 包含文件信息的列表，每个元素是一个字典。
    - current_directory: 源文件所在的目录路径。
    - prefix: 是否为列名添加前缀以避免重复。
    
    返回：
    - 合并后的 DataFrame。
    """
    df_list = []
    
    for source in excel_sources:
        file_path = os.path.join(current_directory, source['file'])
        sheet_name = source['sheet']
        columns = source['columns']
    
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                print(f"文件未找到: {file_path}")
                continue
    
            # 读取指定的工作表和列
            df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)
    
            # 将读取到的 DataFrame 添加到列表中
            df_list.append(df)
    
            print(f"成功读取 {source['file']} 的工作表 {sheet_name}。")
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except ValueError as ve:
            print(f"读取 {source['file']} 的工作表 {sheet_name} 时出错: {ve}")
        except Exception as e:
            print(f"处理 {source['file']} 时发生未知错误: {e}")
    
    if df_list:
        # 检查所有 DataFrame 的行数是否一致
        row_counts = [len(df) for df in df_list]
        if len(set(row_counts)) != 1:
            print("警告：不同文件的数据行数不一致，合并可能导致数据错位或缺失。")
    
        # 横向合并所有 DataFrame
        combined_df = pd.concat(df_list, axis=1)
        return combined_df
    else:
        print("没有数据被读取，未生成合并文件。")
        return None

def main():
    # 获取当前工作目录路径
    # current_directory = os.getcwd()
    # print(current_directory)
    current_directory = './input_data'
    current_directory_copy = './output_data'
    # 定义输出的合并后 Excel 文件的路径
    output_file = os.path.join(current_directory_copy, 'Total_data.xlsx')

    # output_file = os.path.join(current_directory, 'Total_data.xlsx')
    
    # 定义要保存到的工作表名称
    sheet_names = ['PPD', 'Temperature', 'Setpoint','Day_energy','Total_energy', 'Total_PPD']


    Day_sheet = 'Day_42'#修改这个地方来更改具体的天数


    # 定义源文件的信息 - 第一组列，保存到 'Combined_Data1'
    PPD_sheet = [
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_LAG_PPD1', 'MADDPG_LAG_PPD2', 'MADDPG_LAG_PPD3', 'MADDPG_LAG_PPD4', 'MADDPG_LAG_PPD5']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_Lag_PPD1', 'MASAC_Lag_PPD2', 'MASAC_Lag_PPD3', 'MASAC_Lag_PPD4', 'MASAC_Lag_PPD5']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_PPD1', 'MADDPG_PPD2', 'MADDPG_PPD3', 'MADDPG_PPD4', 'MADDPG_PPD5']
        },
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['DCMASAC_PPD1', 'DCMASAC_PPD2', 'DCMASAC_PPD3', 'DCMASAC_PPD4', 'DCMASAC_PPD5']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_People1', 'MASAC_PPD1', 'MASAC_PPD2', 'MASAC_PPD3', 'MASAC_PPD4', 'MASAC_PPD5']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['IPO_PPD1', 'IPO_PPD2', 'IPO_PPD3', 'IPO_PPD4', 'IPO_PPD5']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['P3O_PPD1', 'P3O_PPD2', 'P3O_PPD3', 'P3O_PPD4', 'P3O_PPD5']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['Rule_PPD_1', 'Rule_PPD_2', 'Rule_PPD_3', 'Rule_PPD_4', 'Rule_PPD_5']
        },
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': Day_sheet,
        #     'columns': ['MPC_PPD_1', 'MPC_PPD_2', 'MPC_PPD_3', 'MPC_PPD_4', 'MPC_PPD_5']
        # },
        # 可以继续添加更多源文件的信息
    ]
    # 定义文件及其列名生成规则
    # 定义源文件的信息 - 第二组列，保存到 'Combined_Data2'
    Temperature_sources_sheet = [
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_LAG_temperature1', 'MADDPG_LAG_temperature2','MADDPG_LAG_temperature3', 'MADDPG_LAG_temperature4','MADDPG_LAG_temperature5']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_Lag_temperature1', 'MASAC_Lag_temperature2','MASAC_Lag_temperature3', 'MASAC_Lag_temperature4','MASAC_Lag_temperature5']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_temperature1', 'MADDPG_temperature2','MADDPG_temperature3', 'MADDPG_temperature4','MADDPG_temperature5']
        },
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['DCMASAC_temperature1', 'DCMASAC_temperature2','DCMASAC_temperature3', 'DCMASAC_temperature4','DCMASAC_temperature5']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_People1', 'MASAC_temperature1', 'MASAC_temperature2', 'MASAC_temperature3', 'MASAC_temperature4', 'MASAC_temperature5']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['IPO_temperature1', 'IPO_temperature2','IPO_temperature3', 'IPO_temperature4','IPO_temperature5']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['P3O_temperature1', 'P3O_temperature2','P3O_temperature3', 'P3O_temperature4','P3O_temperature5']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['Rule_temperature1', 'Rule_temperature2','Rule_temperature3', 'Rule_temperature4','Rule_temperature5']
        }
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': Day_sheet,
        #     'columns': ['MPC_temperature1', 'MPC_temperature2','MPC_temperature3', 'MPC_temperature4','MPC_temperature5']
        # },
        # 可以继续添加更多源文件的信息
    ]
    setpoint_sources_sheet = [
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_LAG_People1', 'MADDPG_LAG_heating1', 'MADDPG_LAG_heating2','MADDPG_LAG_heating3', 'MADDPG_LAG_heating4','MADDPG_LAG_heating5'
                        ,'MADDPG_LAG_cooling1', 'MADDPG_LAG_cooling2','MADDPG_LAG_cooling3', 'MADDPG_LAG_cooling4','MADDPG_LAG_cooling5']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_Lag_heating1', 'MASAC_Lag_heating2','MASAC_Lag_heating3', 'MASAC_Lag_heating4','MASAC_Lag_heating5'
                        ,'MASAC_Lag_cooling1', 'MASAC_Lag_cooling2','MASAC_Lag_cooling3', 'MASAC_Lag_cooling4','MASAC_Lag_cooling5']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_heating1', 'MADDPG_heating2','MADDPG_heating3', 'MADDPG_heating4','MADDPG_heating5',
                        'MADDPG_cooling1', 'MADDPG_cooling2','MADDPG_cooling3', 'MADDPG_cooling4','MADDPG_cooling5']
        },
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['DCMASAC_heating1', 'DCMASAC_heating2','DCMASAC_heating3', 'DCMASAC_heating4','DCMASAC_heating5',
                        'DCMASAC_cooling1', 'DCMASAC_cooling2','DCMASAC_cooling3', 'DCMASAC_cooling4','DCMASAC_cooling5']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_People1', 'MASAC_heating1', 'MASAC_heating2','MASAC_heating3', 'MASAC_heating4','MASAC_heating5'
                        ,'MASAC_cooling1', 'MASAC_cooling2','MASAC_cooling3', 'MASAC_cooling4','MASAC_cooling5']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['IPO_heating1', 'IPO_heating2','IPO_heating3', 'IPO_heating4','IPO_heating5',
                        'IPO_cooling1', 'IPO_cooling2','IPO_cooling3', 'IPO_cooling4','IPO_cooling5']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['P3O_heating1', 'P3O_heating2','P3O_heating3', 'P3O_heating4','P3O_heating5',
                        'P3O_cooling1', 'P3O_cooling2','P3O_cooling3', 'P3O_cooling4','P3O_cooling5']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['Rule_heating1', 'Rule_heating2','Rule_heating3', 'Rule_heating4','Rule_heating5',
                        'Rule_cooling1', 'Rule_cooling2','Rule_cooling3', 'Rule_cooling4','Rule_cooling5']
        },
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': Day_sheet,
        #     'columns': ['MPC_heating1', 'MPC_heating2','MPC_heating3', 'MPC_heating4','MPC_heating5',
        #                 'MPC_cooling1', 'MPC_cooling2','MPC_cooling3', 'MPC_cooling4','MPC_cooling5']
        # },
        # 可以继续添加更多源文件的信息
    ]
    Day_Energy_sources_sheet = [
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_LAG_People1', 'MADDPG_LAG_energy']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_Lag_energy']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MADDPG_energy']
        },
        
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['DCMASAC_energy']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['MASAC_energy']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['IPO_energy']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['P3O_energy']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': Day_sheet,
            'columns': ['Rule_energy']
        },
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': Day_sheet,
        #     'columns': ['MPC_energy']
        # },
        # 可以继续添加更多源文件的信息
    ]
    Total_energy_sources_sheet = [ 
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MADDPG_LAG_energy']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MASAC_Lag_energy']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MADDPG_energy']
        },
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': 'All_data',
            'columns': ['DCMASAC_energy']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MASAC_energy']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': 'All_data',
            'columns': ['IPO_energy']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': 'All_data',
            'columns': ['P3O_energy']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': 'All_data',
            'columns': ['Rule_energy']
        },
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': 'All_data',
        #     'columns': ['MPC_energy']
        # },

        ]
    
    Total_PPD_sources_sheet = [
        {
            'file': 'MADDPG_LAG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MADDPG_LAG_People1','MADDPG_LAG_PPD1', 'MADDPG_LAG_PPD2', 'MADDPG_LAG_PPD3','MADDPG_LAG_PPD4', 'MADDPG_LAG_PPD5']
        },
        {
            'file': 'MASAC_LAG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MASAC_Lag_PPD1', 'MASAC_Lag_PPD2','MASAC_Lag_PPD3', 'MASAC_Lag_PPD4','MASAC_Lag_PPD5']
        },
        {
            'file': 'MADDPG_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MADDPG_PPD1', 'MADDPG_PPD2','MADDPG_PPD3', 'MADDPG_PPD4','MADDPG_PPD5']
        },
        {
            'file': 'DCMASAC_data.xlsx',
            'sheet': 'All_data',
            'columns': ['DCMASAC_PPD1', 'DCMASAC_PPD2','DCMASAC_PPD3', 'DCMASAC_PPD4','DCMASAC_PPD5']
        },
        {
            'file': 'MASAC_data.xlsx',
            'sheet': 'All_data',
            'columns': ['MASAC_People1','MASAC_PPD1', 'MASAC_PPD2', 'MASAC_PPD3','MASAC_PPD4', 'MASAC_PPD5']
        },
        {
            'file': 'IPO_data.xlsx',
            'sheet': 'All_data',
            'columns': ['IPO_PPD1', 'IPO_PPD2','IPO_PPD3', 'IPO_PPD4','IPO_PPD5']
        },
        {
            'file': 'P3O_data.xlsx',
            'sheet': 'All_data',
            'columns': ['P3O_PPD1', 'P3O_PPD2','P3O_PPD3', 'P3O_PPD4','P3O_PPD5']
        },
        {
            'file': 'Rule_data.xlsx',
            'sheet': 'All_data',
            'columns': ['Rule_PPD_1', 'Rule_PPD_2','Rule_PPD_3', 'Rule_PPD_4','Rule_PPD_5']
        },
        # {
        #     'file': 'MPC_data.xlsx',
        #     'sheet': 'All_data',
        #     'columns': ['MPC_PPD_1', 'MPC_PPD_2','MPC_PPD_3', 'MPC_PPD_4','MPC_PPD_5']
        # },
        # 可以继续添加更多源文件的信息
    ]


    # 读取并合并第一组列数据
    PPD_df = read_and_combine(PPD_sheet, current_directory, prefix=True)
    
    # 读取并合并第二组列数据
    Temperature_df = read_and_combine(Temperature_sources_sheet, current_directory, prefix=True)

    # 读取并合并第二组列数据
    setpoint_df = read_and_combine(setpoint_sources_sheet, current_directory, prefix=True)

    # 读取并合并第二组列数据
    Day_Energy_df = read_and_combine(Day_Energy_sources_sheet, current_directory, prefix=True)

    # 读取并合并第二组列数据
    Total_energy_df = read_and_combine(Total_energy_sources_sheet, current_directory, prefix=True)

    # 读取并合并第二组列数据
    Total_PPD_df = read_and_combine(Total_PPD_sources_sheet, current_directory, prefix=True)
    
    # 使用 ExcelWriter 将多个 DataFrame 写入不同的工作表
    if PPD_df is not None or Temperature_df is not None or setpoint_df is not None or Day_Energy_df is not None or Total_energy_df is not None or Total_PPD_df is not None:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                if PPD_df is not None:
                    PPD_df.to_excel(writer, sheet_name=sheet_names[0], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[0]}' 工作表中。")
                
                if Temperature_df is not None:
                    Temperature_df.to_excel(writer, sheet_name=sheet_names[1], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[1]}' 工作表中。")

                if setpoint_df is not None:
                    setpoint_df.to_excel(writer, sheet_name=sheet_names[2], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[2]}' 工作表中。")
                
                if Day_Energy_df is not None:
                    Day_Energy_df.to_excel(writer, sheet_name=sheet_names[3], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[3]}' 工作表中。")

                if Total_energy_df is not None:
                    Total_energy_df.to_excel(writer, sheet_name=sheet_names[4], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[4]}' 工作表中。")

                if Total_PPD_df is not None:
                    Total_PPD_df.to_excel(writer, sheet_name=sheet_names[5], index=False)
                    print(f"数据已成功合并并保存到 {output_file} 的 '{sheet_names[5]}' 工作表中。")
                    
            print(f"所有数据已成功保存到 {output_file}。")
        except Exception as e:
            print(f"保存合并后的数据到 {output_file} 时出错: {e}")
    else:
        print("没有数据被读取，未生成合并文件。")

if __name__ == "__main__":
    main()
