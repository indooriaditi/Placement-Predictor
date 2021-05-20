import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import os

#students data
df=pd.read_csv(os.path.abspath('data/10yrsmerged.csv'))

branches=df['Branch'].unique()

fig=px.bar(df.groupby(['Branch','Gender']).agg({'Gender':np.count_nonzero}).rename(columns={'Gender':'Count'}).reset_index(), x='Branch',y='Count',color='Gender',barmode="group")
fig1=px.bar(df.groupby(['Branch','Gender']).agg({'CGPA':np.nanmean}).rename(columns={'CGPA':'Average CGPA'}).reset_index(),x='Branch',y='Average CGPA',color='Gender',barmode="group")
fig2=px.bar(df.groupby('Branch').agg({'CGPA':np.nanmean}).reset_index().sort_values(['CGPA'],ascending=False),x='Branch',y='CGPA',color='Branch',hover_name='CGPA')
fig3=px.sunburst(df,path=['Branch','Gender'],values='No. of Placements',color='Branch',hover_name='No. of Placements')

yrs=df['Year'].unique()

vis3=df[df['Status']=='Placed'].groupby('Year').agg({'No. of Placements':np.nansum,'Status':np.count_nonzero}).rename(columns={'No. of Placements':'Total No. of Offer Letters','Status':'No. of Placed Students'}).reset_index()
vis3=pd.melt(vis3,id_vars=['Year'],value_vars=['Total No. of Offer Letters','No. of Placed Students']).rename(columns={'value':'count'})
fig4=px.line(vis3,x='Year',y='count',color='variable')

#companies data
pm=pd.read_csv(os.path.abspath('data/11yrsmerged_recruitmentwise.csv'))
pm.set_index('Name of the Organization',inplace=True)

br_tot=pd.DataFrame(pm.loc['GRAND TOTAL'])
br_tot=br_tot.drop('TOTAL').sort_values(by='GRAND TOTAL',ascending=False)
figg=px.bar(br_tot)

pm1=pm[~(pm.index=='GRAND TOTAL')]

cols=['ECE', 'CSE', 'EEE', 'IT', 'MECH', 'PROD', 'CIVIL', 'CHEM', 'BIO- TECH',
       'MCA', 'MBA', 'TOTAL']

pm2=pm1.drop('TOTAL',axis=1)

ttcomp=pm1.sort_values(by='TOTAL',ascending=False).head(10).index

#salary data
conca=pd.read_csv(os.path.abspath('data/11yrsmerged_salarywise_withinternships.csv'))
PL=pd.read_csv(os.path.abspath('data/11yrsmerged_salarywise_withoutinternships.csv'))


sl=PL.groupby('Year').agg({'Salary Per Annum (Rs. In Lakhs)':[np.mean,np.median]}).stack().reset_index().rename(columns={'level_1':'Type'})
avgpkg=px.line(sl,x='Year',y='Salary Per Annum (Rs. In Lakhs)',color='Type')
avgpkg1=px.box(sl,x='Type',y='Salary Per Annum (Rs. In Lakhs)')


sal11=PL.groupby('Year').agg({'Salary Per Annum (Rs. In Lakhs)':[np.mean,np.median]})
mean_pkg=sal11['Salary Per Annum (Rs. In Lakhs)']['mean'].mean()
median_pkg=sal11['Salary Per Annum (Rs. In Lakhs)']['median'].median()

years=['Overall',10,11,12,13,14,15,16,17,18,19,20,21]


t25=pm1.sort_values(by='TOTAL',ascending=False).head(25)
t25=t25.reset_index()
t25['Type']=np.nan
lst=['MICROSOFT','ORACLE','NCR CORPORATION','MICRONTECHNOLOGY',"BYJU'S",'PEGA SYSTEMS']
lst1=['JPMC','BANK OF AMERICA']

def type(org):
    if org in lst:
        return 'Product-Based'
    elif org in lst1:
        return "Finance-Based"
    else:
        return 'Service-Based'

t25['Type']=t25['Name of the Organization'].apply(lambda x: type(x))
t25=t25.groupby('Type').agg({'TOTAL':np.nansum}).reset_index()
t25['TOTAL']=t25['TOTAL'].astype('int')
spf=px.bar(t25,x='Type',y='TOTAL')



def create_dash_application(flask_app):
    
    dashapp=dash.Dash(server=flask_app,name='Dashboard',url_base_pathname='/dash/')

    dashapp.layout = html.Div([

        html.A('Go back to Predictor', href="/"),
        html.Br(),
    
        html.Div([
            html.H1('DASHBOARD')
        ],
        style={'textAlign': 'center'}),
        
        html.Br(),
        
        html.Div([
            html.H2('Branch Wise No. of Placements')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='bnpl',
            figure=figg
        ),
        
    
        
        html.Div([
            html.H2('Top companies across Branches (Recruitment-Wise)')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='topten',
                    options=[{'label': i, 'value': i} for i in cols],
                    value='TOTAL'
                )  
            ]),
        dcc.Graph(id='tt'),
        
    
        
        html.Div([
            html.H2('Top Ten companies Branch-Wise Recruiment')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='toptencomp',
                    options=[{'label': i, 'value': i} for i in ttcomp],
                    value='COGNIZANT'
                )  
            ]),
        dcc.Graph(id='ttcomp'),
        
    
        
        html.Div([
            html.H2('Top Ten companies Salary-Wise')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='years',
                    options=[{'label': i, 'value': i} for i in years],
                    value='Overall'
                )  
            ]),
        dcc.Graph(id='ttsal'),
        

        
        html.Div([
            html.H2('Increase in Average Salary Package over the Years')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='avgpkg',
            figure=avgpkg
        ),
        
    
        
        html.Div([
            html.H2('Analysis of Average Salary Package')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='avgpkg1',
            figure=avgpkg1
        ),
        html.Div([
            html.P(children=["Average Package of 11 years is: ", mean_pkg]),
            html.P(children=["Median Package of 11 years is: ", median_pkg])
        ],
        style={'textAlign': 'center'}),
        
    
        
        html.Div([
            html.H2('Ratio of recruitments in Service-based, Product-based & Finance-based Companies')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='spf',
            figure=spf
        ),
        
        
        html.Div([
            html.H2('Ratio of Placed VS Unplaced Students')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='yrs',
                    options=[{'label': i, 'value': i} for i in yrs],
                    value=21
                )  
            ]),
        dcc.Graph(id='plunpl'),
        
        
        html.Div([
            html.H2('Total no. of Offer Letters VS No. of Placed Students')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='yrs1',
                    options=[{'label': i, 'value': i} for i in yrs],
                    value=21
                )  
            ]),
        dcc.Graph(id='offl'),
        

        html.Div([
            html.H2('Change in Total no. of Offer Letters and No. of Placed Students over the years')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='lineplot',
            figure=fig4
        ),
        
        
        html.Div([
            html.H2('Increase in the number of Placements over the years')
        ],
        style={'textAlign': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='branch',
                    options=[{'label': i, 'value': i} for i in branches],
                    value='IT'
                )  
            ]),
        dcc.Graph(id='increase'),
        
    
        
        html.Div([
            html.H2('Branch-Wise Gender Ratio')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='genderratio',
            figure=fig
        ),
        

        
        html.Div([
            html.H2('CGPA vs Placements')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            id='year-slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            value=df['Year'].min(),
            marks={str(year): str(year) for year in df['Year'].unique()},
            step=None
        ),
        
    
        
        html.Div([
            html.H2('Branch-Wise Average CGPA')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='avgcgpa',
            figure=fig2
        ),
        
    
        
        html.Div([
            html.H2('Branch-Wise & Gender-Wise Average CGPA')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='gavgcgpa',
            figure=fig1
        ),
        
    
        
        html.Div([
            html.H2('Branch Wise Comparision of No. of Placements with respect to Gender')
        ],
        style={'textAlign': 'center'}),
        dcc.Graph(
            id='sunburst',
            figure=fig3
        )
    ])


    

    @dashapp.callback(
        Output('tt', 'figure'),
        Input('topten', 'value'))
    def update_fig(selected_value):
        filtered_df = pm1.sort_values(by=selected_value, ascending=False).head(10)[[selected_value]].reset_index()

        figg = px.bar(filtered_df, x='Name of the Organization', y=selected_value)

        figg.update_layout(transition_duration=500)

        return figg

    @dashapp.callback(
        Output('ttcomp', 'figure'),
        Input('toptencomp', 'value'))
    def update_figtt(selected_comp):
        
        filtered_df = pd.DataFrame(pm2.loc[selected_comp])
        filtered_df = filtered_df.rename(columns={selected_comp :'No. of recruitments'}).sort_values(by='No. of recruitments', ascending=False)
        filtered_df = filtered_df[(filtered_df.T != 0).any()].reset_index().rename(columns={'index':'Branch'})

        figg1 = px.bar(filtered_df, x='Branch',y='No. of recruitments')

        figg1.update_layout(transition_duration=500)

        return figg1

    @dashapp.callback(
        Output('ttsal', 'figure'),
        Input('years', 'value'))
    def update_figttsal(selectedyear):
        
        if selectedyear=='Overall':
            PL_all=PL.sort_values(by=['Salary Per Annum (Rs. In Lakhs)','TOTAL'],ascending=False).head(10)[['Name of the Organization','Salary Per Annum (Rs. In Lakhs)','Year']]
            ttsal=px.bar(PL_all,x='Name of the Organization',y='Salary Per Annum (Rs. In Lakhs)')
        else:
            ps=PL[PL['Year']==selectedyear].sort_values(by=['Salary Per Annum (Rs. In Lakhs)','TOTAL'],ascending=False).head(10)[['Name of the Organization','Salary Per Annum (Rs. In Lakhs)']]
            ttsal=px.bar(ps,x='Name of the Organization',y='Salary Per Annum (Rs. In Lakhs)')

        ttsal.update_layout(transition_duration=200)
        return ttsal

    @dashapp.callback(
        Output('plunpl', 'figure'),
        Input('yrs', 'value'))
    def update_figplunpl(selectedyear):
        
        vis=df[df['Year']==selectedyear].groupby(['Branch','Status']).agg({'Status':np.count_nonzero}).rename(columns={'Status':'Count'}).reset_index().sort_values(by=['Status','Count'],ascending=False)
        plunplfig=px.bar(vis,x='Branch',y='Count',color='Status')
        
        plunplfig.update_layout(transition_duration=200)
        return plunplfig

    @dashapp.callback(
        Output('offl', 'figure'),
        Input('yrs1', 'value'))
    def update_figplunpl(selectedyear):
        
        vis1=df[(df['Year']==selectedyear)&(df['Status']=='Placed')].groupby('Branch').agg({'No. of Placements':np.nansum,'Status':np.count_nonzero}).rename(columns={'No. of Placements':'Total No. of Offer Letters','Status':'No. of Placed Students'}).reset_index()
        vis1=pd.melt(vis1,id_vars=['Branch'],value_vars=['Total No. of Offer Letters','No. of Placed Students']).rename(columns={'value':'count'}).sort_values(by='count',ascending=False)
        offlfig=px.bar(vis1,x='Branch',y='count',color='variable',barmode='group')
        
        
        offlfig.update_layout(transition_duration=200)
        return offlfig

    @dashapp.callback(
        Output('increase', 'figure'),
        Input('branch', 'value'))
    def update_figure(selected_branch):
        filtered_df = df[df.Branch == selected_branch]

        fig = px.line(filtered_df.groupby('Year').sum().reset_index(), x="Year", y="No. of Placements")

        fig.update_layout(transition_duration=500)

        return fig

    @dashapp.callback(
        Output('graph-with-slider', 'figure'),
        Input('year-slider', 'value'))
    def update_figure1(selected_year):
        filtered_df = df[df.Year == selected_year]

        fig1 = px.scatter(filtered_df, x="CGPA", y="No. of Placements",
                        color="CGPA", hover_name="CGPA",
                        log_x=True, size_max=55)

        fig1.update_layout(transition_duration=200)

        return fig1

    return dashapp