import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# dataloading
df = pd.read_csv('Results_21Mar2022.csv')

# preprocessing
def preprocess_data(df):
    # Unify the names of the eating groups
    df['diet_group'] = (
        df['diet_group']
        .str.lower()
        .str.replace(' ', '_')
        .replace({
            'meat': 'meat_100+',
            'meat50': 'meat_50-99',
            'meat<50': 'meat_<50',
            'meat_50': 'meat_50-99',  # replace another possible features
            'meat100': 'meat_100+'
        })
    )
    # 新增饮食组数值编码
    df['diet_code'] = df['diet_group'].astype('category').cat.codes
    # Order age
    # 处理年龄组
    age_order = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    df['age_group'] = pd.Categorical(df['age_group'], categories=age_order, ordered=True)
    
    # 新增数值型年龄字段（强制转换为整数）
    age_bins = {
        '20-29': 25, '30-39': 35, '40-49': 45,
        '50-59': 55, '60-69': 65, '70-79': 75
    }
    df['age_numeric'] = df['age_group'].map(age_bins).astype(int)  # 关键修复点
    
    # 确保无缺失值
    if df['age_numeric'].isnull().any():
        raise ValueError("存在未处理的年龄组: {}".format(
            df.loc[df['age_numeric'].isnull(), 'age_group'].unique()
        ))

# 验证字段类型
    assert pd.api.types.is_integer_dtype(df['age_numeric']), "age_numeric 必须是整数类型"
    # Key Indicator Selection
    metrics = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
    return df[['diet_group', 'sex', 'age_group', 'n_participants'] + metrics]

df_clean = preprocess_data(df)
diets = df['diet_group'].unique()
metrics = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']

# 验证字段类型
assert pd.api.types.is_integer_dtype(df['age_numeric']), "age_numeric 必须是整数类型"

# 初始化Dash应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Diet Impact Explorer", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Dropdown(
            id='diet-selector',
            options=[{'label': d, 'value': d} for d in df['diet_group'].unique()],
            value=[df['diet_group'].unique()[0]],
            multi=True
        )
    ], style={'width': '60%','margin': 'auto'}),
    
    html.Div(style={'height': '60px'}),  # 插入50px高的空div作为间隔
    
    dcc.Graph(
        id='main-plot',style={
        'height': '600px',  # 视口高度的60%
        'width': '90%',    # 父容器宽度的90%
        'margin': '20px auto'   # 水平居中
        },
    config={'responsive': True}  # 响应式布局
    ),
    
    dcc.Slider(
        id='age-slider',
        min=20,
        max=80,
        step=10,
        value=30,
        marks={age: str(age) for age in range(20, 80, 10)},
        tooltip={'placement': 'bottom'},
    )
])
@app.callback(
    Output('main-plot', 'figure'),
    [Input('diet-selector', 'value'),
     Input('age-slider', 'value')]
)
def update_plot(selected_diets, selected_age):
    filtered = df[
        (df['diet_group'].isin(selected_diets)) & 
        (df['age_numeric'] >= selected_age) & 
        (df['age_numeric'] < selected_age + 10)
    ]
    
    if filtered.empty:
        return px.parallel_coordinates(pd.DataFrame(), title="No Data")
    
    # 创建数值到分类的映射字典
    diet_mapping = dict(enumerate(filtered['diet_group'].astype('category').cat.categories))
    # 检查每个维度的数据长度
    columns_to_check = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
    for col in columns_to_check:
        print(f"Column {col} length: {len(filtered[col])}")
    # 生成图形
    fig = px.parallel_coordinates(
        filtered,
        color="diet_code",
        dimensions=['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut'],
        color_continuous_scale=px.colors.qualitative.Pastel,
        labels={m: m.replace('_', ' ').upper() for m in metrics}
    )
    
    # 更新布局配置颜色轴
    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Diet Group',
            tickvals=list(diet_mapping.keys()),  # 数值编码
            ticktext=list(diet_mapping.values()) # 实际分类名称
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='white',
        margin={'t': 40}
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)