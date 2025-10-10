import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as componentspip
import os
import tempfile
from report_generator import generate_demand_letter_from_text, create_demand_package_final_reports, create_internal_final_reports
import base64
from llm_processing import OPENAI_API_KEY, llm
import shutil
import time


# -------------------- Login Credential System --------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "exluser": "exl2025"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def login():
    st.image("exl logo.png", use_container_width=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- App Config and Style --------------------
st.set_page_config(page_title="Litigation Dashboard", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; color: black; }
        section[data-testid="stSidebar"] { background-color: #F5F5F5 !important; color: black !important; }
        * { color: black !important; }
        div[data-baseweb="select"], div[data-baseweb="popover"], div[data-baseweb="option"], div[data-baseweb="menu"] {
            background-color: white !important; color: black !important; border: 1px solid #ccc !important; border-radius: 5px !important;
        }
        div[data-baseweb="option"]:hover, div[data-baseweb="option"][aria-selected="true"] {
            background-color: #e6e6e6 !important;
        }
        .stButton > button {
            background-color: white !important; color: black !important; border: 1px solid #ccc !important; border-radius: 5px !important;
        }
        .stButton > button:hover {
            background-color: #e6e6e6 !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Allow full page scroll instead of clipped layout */
html, body, [class*="css"]  {
    height: auto !important;
    overflow: auto !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("exl logo.png", use_container_width=True)
    selected_screen = st.radio("📁 Navigation", [
        "📊 Model Recommendations", 
        "📊 Reviewed Claims",
        #"📑 Subrogation Workbench",
        #"🧠 Q&A Assistant", 
        "📊 Monitoring Dashboard",
        #"📈 Litigation KPIs"
    ])


# -------------------- Load Data --------------------
# data_path = 'claims_only_Data.csv'
#data_path = "Syntheticdataset_litigation.csv"
data_path = "merged_file_1.csv"
Notes_path = "Notes1.csv"
@st.cache_data(ttl=0)
def load_data():
    cdf = pd.read_csv(data_path)
    ndf = pd.read_csv(Notes_path,sep =',')
    df = cdf.merge(ndf[['Claim_Number','Claims_Notes','Summary']],how='left',on= 'Claim_Number')
    

    df['ML_SCORE'] = round(df['ML_SCORE'], 2) 
   
    if 'User_Action' not in df.columns:
        df['User_Action'] = ''
    if 'User_Action_Details' not in df.columns:
        df['User_Action_Details'] = ''
    return df

df = load_data()
#df = df[df['Reviewed']==0]
action_detail_mapping = {
                    "Awaiting Additional Info":["Choose an option","Field Report","Inspection Report","Medical Report","Police Report","Property Images","Pending Demand","Pending Counter to Offer"],
                    "Dismiss":["Choose an option","Adequate Reserve","Already Settled","Already with Expert","Already with LL/MCU","Already with SIU","Improper Alert","Arbitration for UM/UIM in this State","ADR"],
                    "Engage an Expert":[],
                    "Increase Reserve":[],
                    "Refer to Large Loss":[],
                    "Refer to SIU":[],
                    "Settle":[],
                    "Additional Authority Granted":[],
                    "In Litigation":["Transfer Notice to New UM"],
                    "In Negotiations":[],
                    "Requesting Mediation":[]
                }
# # Directories you want to clear
# DIR1 = "processed_claims"
# DIR2 = "uploaded_claims"

# def clear_directory(dir_path):
#     """Remove all files and subfolders inside a directory."""
#     if os.path.exists(dir_path):
#         for item in os.listdir(dir_path):
#             item_path = os.path.join(dir_path, item)
#             try:
#                 if os.path.isfile(item_path) or os.path.islink(item_path):
#                     os.remove(item_path)
#                 elif os.path.isdir(item_path):
#                     shutil.rmtree(item_path)
#             except Exception as e:
#                 st.error(f"Error deleting {item_path}: {e}")

# # Sidebar Reset Button
# st.sidebar.subheader("⚙️ Settings")
# if st.sidebar.button("🔄 Reset App"):
#     clear_directory(DIR1)
#     clear_directory(DIR2)
#     st.sidebar.success("✅ All data cleared from directories!")
#     st.rerun()   # refresh app after reset



# -------------------- 📊 Dashboard Screen --------------------

if selected_screen == "📊 Model Recommendations":
    st.title("Litigation Propensity Claims Dashboard")
    df = load_data()
    df = df[df['Reviewed']==0]
    # # Toggle filters
    st.markdown("""
            <style>
            div[data-testid="stCheckbox"] {
                display: flex;
                height: 80px; 
                flex-direction: row; 
                white-space: nowrap; 
                font-size: 20px;  
                           
            }

            </style>
            """, unsafe_allow_html=True)
    enable_filters = st.checkbox("🔎 Enable Filters", value=True)
    claim_search = st.text_input("🔍 Search by Claim Number", key="claim_search")
    if enable_filters:
        st.markdown("""
        <div style='text-align:left; height: 80px; margin-bottom: 6px;font-size: 24px'>
            <b>🛠️ Apply Filters</b>
        </div>""", unsafe_allow_html=True)
        filter_cols = st.columns(2)
        
        with filter_cols[0]:
            peril_filter = st.selectbox("INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

        with filter_cols[1]:
            sub_det = st.selectbox("LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')

        # Apply filters
        filtered_df = df.copy()
        if peril_filter != " ":
            filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
        if sub_det != " ":
            filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]
    else:
        filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['ML_SCORE'],ascending=False)

    # Download filtered suspicious claims
    if not suspicious_df.empty:
        download_df = suspicious_df.copy()
        download_csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=download_csv,
            file_name="suspicious_claims.csv",
            mime="text/csv"
        )

    if suspicious_df.empty:
        st.info("⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("📋 Review and Act on Each Suspected Claim")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns([2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8,4.0,4.0,3.0])

            with cols[0]: st.markdown(f"**Claim:** {row['Claim_Number']}")
            with cols[1]: st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]: st.markdown(f"**State:** {row['FTR_JRSDTN_ST_ABBR']}")
            #with cols[3]: st.markdown(f"**Paid:** ${row['PAID_FINAL']:.2f}")
            #with cols[3]: st.markdown(f"**Accident City:** {row['ACDNT_CITY']}")
            with cols[3]: st.markdown(f"**Body Part Injured:** {row['BODY_PART_INJD_DESC']}")
            with cols[4]: st.markdown(f"**Loss Party:** {row['LOSS_PARTY']}")
            #with cols[7]: st.markdown(f"**Severity:** {row['CLM_LOSS_SEVERITY_CD']}")
            with cols[5]: st.markdown(f"**ML Score:** {row['ML_SCORE']}")

            # --- New Column for Notes Summary Toggle ---
            with cols[6]:
                st.markdown("**Notes**") 
                st.markdown(
                        """
                        <style>
                        div[data-testid="stCheckbox"] {
                            margin-top: -46px;
                            margin-left: 6px;
                            width: 120px !important;
                            height: 50px !important;
                            display: flex;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                        )
                

                show_summary = st.toggle("", key=f"notes_toggle_{idx}")
            # if show_summary:
            #     st.text_area(
            #         "Claim Notes Summary",
            #         value=row.get("Summary", "No summary available."),
            #         height=500,
            #         key=f"notes_area_{idx}"
            #     )
            if show_summary:
                with st.spinner("Generating summary using LLM..."):
                # Pass the claim notes (or whatever column contains raw notes)
                    summary_text = llm(row['Claims_Notes'])
            
                    st.text_area(
                        "Claim Notes Summary",
                        value=summary_text,
                        height=500,
                        key=f"notes_area_{idx}"
                    )




            with cols[7]:
                st.markdown("""
        <div style='text-align:center; height: 30px; margin-bottom: 4px;'>
            <b>Action</b>
        </div>
        """, unsafe_allow_html=True)
                st.markdown("""
        <style>
        div[data-testid="stSelectbox"]{
            display: flex;
            flex-direction: column;
            margin-top: -50px;
            align-items: center;      /* Centers horizontally */
            justify-content: center;
       
        }
       
        </style>
        """, unsafe_allow_html=True)
                action_options = [
    "",
    "Awaiting Additional Info",
    "Dismiss",
    "Engage an Expert",
    "Increase Reserve",
    "Refer to Large Loss",
    "Refer to SIU",
    "Settle",
    "Additional Authority Granted",
    "In Litigation",
    "In Negotiations",
    "Requesting Mediation"
]
               
                # Get current user action safely
                user_action_value = row.get("User_Action")
                # Show the prefilled suggestion (grey hint text)
                # Determine default index safely
                if user_action_value in action_options:
                    default_index = action_options.index(user_action_value)
                else:
                    default_index = 0  # fallback to blank if invalid or missing
               
                # Render dropdown
                selected_action = st.selectbox(
                    "",
                    action_options,
                    key=f"action_{idx}",
                    index=default_index
    )
                # selected_action = st.selectbox(
                #     "",
                #     ["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"],
                #     key=f"action_{idx}",
                #     index=["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"].index(row['User_Action']) if row['User_Action'] in ["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"] else 0
                # )
            

                
                # --- ACTION DETAILS dropdown (depends on Action) ---
                with cols[8]:
    #  Center the label properly
                            st.markdown("""
        <div style='text-align:center; height: 30px; margin-bottom: 4px;'>
            <b>Action Details</b>
        </div>
        """, unsafe_allow_html=True)
    
    # Add CSS once (to center dropdown itself)
                            st.markdown("""
        <style>
        div[data-testid="stSelectbox"]{
            display: flex;
            flex-direction: column;
            margin-top: -50px;
            align-items: center;      /* Centers horizontally */
            justify-content: center;
        }
        
        </style>
        """, unsafe_allow_html=True)
                            if selected_action:
                                action_details_options = action_detail_mapping.get(selected_action, [])
                                
                                # Always include an empty first option
                                options = [] + action_details_options
                                
                                # Determine the default index safely
                                if row.get('User_Action_Details') in action_details_options:
                                    default_index = options.index(row['User_Action_Details'])
                                else:
                                    default_index = 0  # default to blank

                                # Ensure index is within range
                                if default_index < 0 or default_index >= len(options):
                                    default_index = 0

                                # Render selectbox
                                selected_detail = st.selectbox(
                                    "",  # label hidden (custom label above)
                                    options,
                                    key=f"action_details_{idx}",
                                    index=default_index
                                )

                            else:
                                # If no Action selected, show disabled dropdown
                                selected_detail = st.selectbox(
                                    "",
                                    [""],
                                    key=f"action_details_{idx}_disabled"
                                )

                            # #  Dropdown logic
                            # if selected_action:
                            #     action_details_options = action_detail_mapping.get(selected_action, [])
                            #     selected_detail = st.selectbox(
                            #         "",   # label hidden (since we have custom label above)
                            #         [] + action_details_options,
                            #         key=f"action_details_{idx}",
                            #         index=([""] + action_details_options).index(row['User_Action_Details'])
                            #         if row['User_Action_Details'] in action_details_options else 0
                            #     )
                            # else:
                            #     # If no Action selected, disable dropdown
                            #     selected_detail = st.selectbox(
                            #         "",
                            #         [""],
                            #         key=f"action_details_{idx}_disabled"
                            #     )
    #         # with cols[8]:
            #     st.markdown("""
            #         <div style= text-align:center; height: 30px;'>
            #             <b>Action Details</b>
            #         </div>
            #         """, unsafe_allow_html=True)
            #     if selected_action:
            #         action_details_options = action_detail_mapping.get(selected_action, [])
            #         selected_detail = st.selectbox(
            #             "Action Detail" ,
            #             [""] + action_details_options,
            #             key=f"action_details_{idx}",
            #             index=([""] + action_details_options).index(row['User_Action_Details'])
            #             if row['User_Action_Details'] in action_details_options else 0
            #         )
            #     else:
            #         # If no Action selected, disable the details dropdown
            #         selected_detail = st.selectbox(
            #             "Action Details",
            #             [""],
            #             key=f"action_details_{idx}_disabled"
            #         )


            with cols[9]:
                st.markdown("""
                    <div style= text-align:center; height: 30px;'>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)
                #st.markdown("**Save**")
                st.markdown(
                        """
                        <style>
                        div.stButton > button {
                            margin-top: -15px;
                            margin-left: 8px;
                            width: 85px;
                            height: 40px;
                            display: flex;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                if st.button("💾", key=f"save_{idx}"):
                    df_all = pd.read_csv(data_path)
                    df_all.at[idx, 'User_Action'] = selected_action
                    df_all.at[idx, "User_Action_Details"] = selected_detail
                    df_all.to_csv(data_path, index=False)
                    
                    st.success(f"✅ Action saved for Claim {row['Claim_Number']}")

            with st.container():
                st.markdown(f"""
                <div style='
                    margin-top: 2px;
                    margin-bottom: 10px;
                    padding: 6px 10px;
                    font-size: 13px;
                    background-color: #f9f9f9;
                    color: #444;
                    border-radius: 4px;
                '>
                📝
                <i>Confidence score for suggested action:</i> {row['confidence_score']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; 
                <i>Rationale for suggested Action:</i> {row['rationale']}
                </div>
                """, unsafe_allow_html=True)

# -------------------- 📊 Reviewed Claims Screen --------------------

if selected_screen == "📊 Reviewed Claims":
    st.title("Litigation Propensity Claims Dashboard")
    df = load_data()
    df = df[df['Reviewed']==1]
    # Toggle filters
    st.markdown("""
            <style>
            div[data-testid="stCheckbox"] {
                display: flex;
                height: 50px; 
                flex-direction: row; 
                white-space: nowrap;  
                font-size: 20px;               
            }

            </style>
            """, unsafe_allow_html=True)
    enable_filters = st.checkbox("🔎 Enable Filters", value=True)

    claim_search = st.text_input("🔍 Search by Claim Number", key="claim_search")

    if enable_filters:
        st.markdown("""
        <div style='text-align:left; height: 80px; margin-bottom: 6px;font-size: 24px'>
            <b>🛠️ Apply Filters</b>
        </div>""", unsafe_allow_html=True)
        filter_cols = st.columns(2)

        #with filter_cols[0]:
            #state_filter = st.selectbox('STATE', [" "] + list(df['FTR_JRSDTN_ST_ABBR'].unique()), key='state_filter')
        st.markdown("<div id='filter_section'>", unsafe_allow_html=True)
        st.markdown("""
                                    <style>
}                                  
                                    div[data-testid="stSelectbox"] {
                                        
                                        align-items: center;      /* Centers horizontally */
                                        justify-content: center;
                                
                                        
                                    }

                                
                                    </style>
                                    """, unsafe_allow_html=True)
        with filter_cols[0]:
            peril_filter = st.selectbox("INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

        with filter_cols[1]:
            sub_det = st.selectbox("LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')
        st.markdown("</div>", unsafe_allow_html=True)  # close filter_section wrapper

        # Apply filters
        filtered_df = df.copy()
        #if state_filter != " ":
            #filtered_df = filtered_df[filtered_df['FTR_JRSDTN_ST_ABBR'] == state_filter]
        if peril_filter != " ":
            filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
        if sub_det != " ":
            filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]
    else:
        filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['ML_SCORE'],ascending=False)

    # Download filtered suspicious claims
    if not suspicious_df.empty:
        download_df = suspicious_df.copy()
        download_csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=download_csv,
            file_name="suspicious_claims.csv",
            mime="text/csv"
        )

    if suspicious_df.empty:
        st.info("⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("📋 Reviewed Claims")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns([2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8,4.0,4.0,3.0])

            with cols[0]: st.markdown(f"**Claim:** {row['Claim_Number']}")
            with cols[1]: st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]: st.markdown(f"**State:** {row['FTR_JRSDTN_ST_ABBR']}")
            #with cols[3]: st.markdown(f"**Paid:** ${row['PAID_FINAL']:.2f}")
            #with cols[3]: st.markdown(f"**Accident City:** {row['ACDNT_CITY']}")
            with cols[3]: st.markdown(f"**Body Part Injured:** {row['BODY_PART_INJD_DESC']}")
            with cols[4]: st.markdown(f"**Loss Party:** {row['LOSS_PARTY']}")
            #with cols[7]: st.markdown(f"**Severity:** {row['CLM_LOSS_SEVERITY_CD']}")
            with cols[5]: st.markdown(f"**ML Score:** {row['ML_SCORE']}")

            # --- New Column for Notes Summary Toggle ---
            with cols[6]:
                st.markdown("**Notes**")
                st.markdown(
                        """
                        <style>
                        div[data-testid="stCheckbox"] {
                            margin-top: -46px;
                            margin-left: 6px;
                            width: 120px !important;
                            height: 50px !important;
                            display: flex;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                        )
                show_summary = st.toggle("", key=f"notes_toggle_{idx}")
            if show_summary:
                with st.spinner("Generating summary using LLM..."):
                # Pass the claim notes (or whatever column contains raw notes)
                    summary_text = llm(row['Claims_Notes'])
            
                    st.text_area(
                        "Claim Notes Summary",
                        value=summary_text,
                        height=500,
                        key=f"notes_area_{idx}"
                    )
            # if show_summary:
            #     st.text_area(
            #         "Claim Notes Summary",
            #         value=row.get("Summary", "No summary available."),
            #         height=500,
            #         key=f"notes_area_{idx}"
            #     )

            # with cols[7]:
            #     selected_action = st.selectbox(
            #         "Action",
            #         ["", "awaiting additional info", "dismiss", "engage an expert","increase reserve","refer to large loss","refer to SIU","settle","additional authority granted","in litigation","in negotiations","requesting mediation"],
            #         key=f"action_{idx}",
            #         index=["", "awaiting additional info", "dismiss", "engage an expert","increase reserve","refer to large loss","refer to SIU","settle","additional authority granted","in litigation","in negotiations","requesting mediation"].index(row['User_Action']) if row['User_Action'] in ["", "awaiting additional info", "dismiss", "engage an expert","increase reserve","refer to large loss","refer to SIU","settle","additional authority granted","in litigation","in negotiations","requesting mediation"] else 0
            #     )
            # with cols[8]:
            #     # --- ACTION DETAILS dropdown (depends on Action) ---
            #     if selected_action:
            #         action_details_options = action_detail_mapping.get(selected_action, [])
            #         selected_detail = st.selectbox(
            #             f"Action Details",
            #             [""] + action_details_options,
            #             key=f"action_details_{idx}",
            #             index=([""] + action_details_options).index(row['User_Action_Details'])
            #             if row['User_Action_Details'] in action_details_options else 0
            #         )
            #     else:
            #         # If no Action selected, disable the details dropdown
            #         selected_detail = st.selectbox(
            #             f"Action Details",
            #             [""],
            #             key=f"action_details_{idx}_disabled"
            #         )

            
            with cols[7]:

                st.markdown("""
        <div style='text-align:center; height: 30px; margin-bottom: 4px;'>
            <b>Action</b>
        </div>
        """, unsafe_allow_html=True)
                st.markdown("<div id='row_section1'>"
                                , unsafe_allow_html=True)
                st.markdown("""
        <style>
                    
                    div[data-testid="stSelectbox"] {
                        display: flex;
                        flex-direction: column;
                        margin-top: -70px;
                        align-items: center;      /* Centers horizontally */
                        justify-content: center;
                        
        }
    
        </style>
        """, unsafe_allow_html=True)
                action_options = [
    "",
    "Awaiting Additional Info",
    "Dismiss",
    "Engage an Expert",
    "Increase Reserve",
    "Refer to Large Loss",
    "Refer to SIU",
    "Settle",
    "Additional Authority Granted",
    "In Litigation",
    "In Negotiations",
    "Requesting Mediation"
]
            
                # Get current user action safely
                user_action_value = row.get("User_Action")
                # Show the prefilled suggestion (grey hint text)
                # Determine default index safely
                if user_action_value in action_options:
                    default_index = action_options.index(user_action_value)
                else:
                    default_index = 0  # fallback to blank if invalid or missing
            
                # Render dropdown
                selected_action = st.selectbox(
                    "",
                    action_options,
                    key=f"action_{idx}",
                    index=default_index
                    )
                st.markdown("</div>", unsafe_allow_html=True)  # close filter_section wrapper
            # selected_action = st.selectbox(
                #     "",
                #     ["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"],
                #     key=f"action_{idx}",
                #     index=["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"].index(row['User_Action']) if row['User_Action'] in ["", "Awaiting Additional Info", "Dismiss", "Engage an Expert","Increase Reserve","Refer to Large Loss","Refer to SIU","Settle","Additional Authority Granted","In Litigation","In Negotiations","Requesting Mediation"] else 0
                # )
               
                                

                                    
                # --- ACTION DETAILS dropdown (depends on Action) ---
                with cols[8]:
    #  Center the label properly
                            st.markdown("""
        <div style='text-align:center; height: 30px; margin-bottom: 4px;'>
            <b>Action Details</b>
        </div>
        """, unsafe_allow_html=True)
    
    # Add CSS once (to center dropdown itself)
                            st.markdown("<div id='row_section'>", unsafe_allow_html=True)
                            st.markdown("""
                            <style>

                                    div[data-testid="stSelectbox"] {
                                        margin-top: -70px;
                                        flex-direction: column;
                                        
                                        align-items: center;      /* Centers horizontally */
                                        justify-content: center;
                                         
                                    }
                                
                                    </style>
                                    """, unsafe_allow_html=True)
                           

                            if selected_action:
                                action_details_options = action_detail_mapping.get(selected_action, [])
                                
                                # Always include an empty first option
                                options = [] + action_details_options
                                
                                # Determine the default index safely
                                if row.get('User_Action_Details') in action_details_options:
                                    default_index = options.index(row['User_Action_Details'])
                                else:
                                    default_index = 0  # default to blank

                                # Ensure index is within range
                                if default_index < 0 or default_index >= len(options):
                                    default_index = 0

                                # Render selectbox
                                selected_detail = st.selectbox(
                                    "",  # label hidden (custom label above)
                                    options,
                                    key=f"action_details_{idx}",
                                    index=default_index
                                )

                            else:
                                # If no Action selected, show disabled dropdown
                                selected_detail = st.selectbox(
                                    "",
                                    [""],
                                    key=f"action_details_{idx}_disabled"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)  # close filter_section wrapper




            # with cols[9]:
            #     if st.button("💾 Save", key=f"save_{idx}"):
            #         df_all = pd.read_csv(data_path)
            #         df_all.at[idx, 'User_Action'] = selected_action
            #         df_all.at[idx, "User_Action_Details"] = selected_detail
            #         df_all.to_csv(data_path, index=False)
            #         st.success(f"✅ Action saved for Claim {row['Claim_Number']}")
            with cols[9]:
                st.markdown("""
                    <div style= text-align:center; height: 30px;'>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)
                #st.markdown("**Save**")
                st.markdown(
                        """
                        <style>
                        div.stButton > button {
                            margin-top: -15px;
                            margin-left: 8px;
                            width: 85px;
                            height: 40px;
                            display: flex;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                if st.button("💾", key=f"save_{idx}"):
                    df_all = pd.read_csv(data_path)
                    df_all.at[idx, 'User_Action'] = selected_action
                    df_all.at[idx, "User_Action_Details"] = selected_detail
                    df_all.to_csv(data_path, index=False)
                    
                    st.success(f"✅ Action saved for Claim {row['Claim_Number']}")

            with st.container():
                st.markdown(f"""
                <div style='
                    margin-top: 2px;
                    margin-bottom: 10px;
                    padding: 6px 10px;
                    font-size: 13px;
                    background-color: #FFFFFF;
                    color: #444;
                    border-radius: 4px;'>
                📝
                <i>Reviewed 2 days ago</i>
                </div>
                """, unsafe_allow_html=True)



# # # -------------------- 📈 KPI Screen --------------------
# elif selected_screen == "📈 Subrogation KPIs":
#     st.title("📈 Subrogation Business KPIs")
#     st.set_page_config(page_title="Subrogation KPI Dashboard", layout="wide")
#     # Aggregated KPIs
#     total_claims = df["Claim_Number"].nunique()
#     total_paid = df["PAID_FINAL"].sum()
#     total_target_subro = df["Target_Subro"].sum()
#     avg_paid = df["PAID_FINAL"].mean()
#     avg_target_subro = df["Target_Subro"].mean()
#     Total_Recovered = df['RECOVERY_AMT'].sum()
#     AVG_Recovered = df['RECOVERY_AMT'].mean()

#     col1, col2, col3, col4, col5, col6 = st.columns(6)
#     col1.metric("🧾 Total Claims", f"{total_claims}")
#     col2.metric("💰 Total Paid", f"${total_paid:,.0f}")
#     col3.metric("🎯 Claims In Subro", f"{total_target_subro:,.0f}")
#     col4.metric("📉 Avg Paid / Claim", f"${avg_paid:,.0f}")
#     # col5.metric("📈 Avg Target Subro / Claim", f"${avg_target_subro:,.0f}")
#     col5.metric("📈 Total Recovered", f"${Total_Recovered:,.0f}")
#     col6.metric("📈 Avg Recoverd / Claim", f"${AVG_Recovered:,.0f}")


#     st.markdown("---")

#     # Aggregated by Accident State
#     st.subheader("Subrogation KPIs by State")
#     state_summary = df.groupby("STATE_GROUP").agg({
#         "Claim_Number": "count",
#         "PAID_FINAL": "sum",
#         "Target_Subro": "sum"
#     }).reset_index().rename(columns={"Claim_Number": "Total Claims"})

#     fig1 = px.bar(state_summary, x="STATE_GROUP", y="Target_Subro",
#                 title="Target Subrogation by State", labels={"ACDNT_ST_DESC": "State Group"})
#     st.plotly_chart(fig1, use_container_width=True)

#     # Aggregated by Account Category
#     st.subheader("Subrogation KPIs by Account Category")
#     acct_summary = df.groupby("ACCT_CR_DESC").agg({
#         "Claim_Number": "count",
#         "PAID_FINAL": "sum",
#         "Target_Subro": "sum"
#     }).reset_index().rename(columns={"Claim_Number": "Total Claims"})

#     fig2 = px.bar(acct_summary, x="ACCT_CR_DESC", y="Target_Subro",
#                 title="Target Subrogation by Account Category", labels={"ACCT_CR_DESC": "Account Category"})
#     st.plotly_chart(fig2, use_container_width=True)


# -------------------- 📊 Monitoring Dashboard --------------------
elif selected_screen == "📊 Monitoring Dashboard":
    st.title("📊 Monitoring Dashboard - Power BI")

    st.markdown("#### Embedded Power BI Dashboard Below:")
    
    powerbi_embed_url = """
    <iframe title="SUBROGATION PROPENSITY MODEL MONITORING" width="1140" height="600" 
        src="https://app.powerbi.com/reportEmbed?reportId=49d274d9-37a4-4f06-ac05-dc7a98960ed9&autoAuth=true&ctid=dafe49bc-5ac3-4310-97b4-3e44a28cbf18&actionBarEnabled=true" 
        frameborder="0" allowFullScreen="true"></iframe>
    """
    componentspip.html(powerbi_embed_url, height=650)


# # -------------------- 🧠 Q&A Assistant --------------------
# elif selected_screen == "🧠 Q&A Assistant":
#     st.title("🧠 Q&A Assistant (Powered by OpenAI)")
#     st.markdown("This assistant answers questions based on the Adjuster Claim Notes.")

#     # Select claim
#     ndf = df[df['Claims_Notes'].notnull()]
#     selected_claim = st.selectbox("Select a Claim Number", ndf["Claim_Number"].unique())

#     claim_notes = df[df["Claim_Number"] == selected_claim]["Claims_Notes"].values[0] if "Claims_Notes" in df.columns else ""


#     st.markdown("### 📄 Original Claim Notes")
#     st.text_area("Claim Notes", claim_notes, height=200, disabled=True)

#     # --------- Q&A Chatbot Section ----------
#     st.markdown("### 💬 Ask Questions About This Claim")

#     # --- User Question Section ---
#     user_question = st.text_input("Type your question here:")
#     if user_question.strip():
#         with st.spinner("Generating answer..."):
#             try:
#                 qa_prompt = f"""
#                 Use the following claim notes to answer the question.
#                 Claim Notes:
#                 {claim_notes}

#                 Question: {user_question}
#                 Answer based only on the claim notes.
#                 """
#                 # response = client.models.generate_content(
#                 #     model="gemini-2.5-flash",
#                 #     contents=qa_prompt
#                 # )
#                 # answer = response.text if hasattr(response, "text") else "No answer generated."
#                 answer = llm(qa_prompt)

#                 st.markdown("**Answer:**")
#                 st.info(answer)
#             except Exception as e:
#                 st.error(f"Error generating answer: {e}")

#     # --- Template Questions Section ---
#     st.markdown("---")
#     st.markdown("### 📑 Frequently Asked Questions")

#     template_questions = [
#         "Mention all the key people involved in claim",
#         "Mention all the key organisation involved in claim",
#         "What is percentage of fault for insured?",
#         "Was a third party responsible for the loss?",
#         "Are there statute of limitations to consider?",
#         "Is this a Comparative negligence and contributory negligence state?",
#         "Is there clear documentation of how the loss occurred?",
#         "What is the reason or cause of Loss?",
#         "Are there any admissions of fault or liability by another party?",
#         "Is the third party insured, and who is their carrier?",
#         "Was a police report or official investigation conducted?",
#         "What damages were paid out and to whom?",
#         "Were any waivers of subrogation signed or implied?"

#     ]

#     for q in template_questions:
#         with st.expander(q):  # Expandable answers for each template question
#             with st.spinner("Fetching answer..."):
#                 try:
#                     qa_prompt = f"""
#                     You are an insurance claims assistant.
#                     Use the following claim notes to answer the question.
#                     Claim Notes:
#                     {claim_notes}

#                     Question: {q}
#                     Answer based only on the claim notes.
#                     """
#                     answer = llm(qa_prompt)
#                     st.markdown("**Answer:**")
#                     st.success(answer)
#                 except Exception as e:
#                     st.error(f"Error generating answer: {e}")

# # -------------------- 📑 Actioned Claims Screen --------------------
# elif selected_screen == "📑 Subrogation Workbench":
    
#     # Upload & process files
#     UPLOAD_BASE_DIR = "uploaded_claims"
#     PROCESSED_BASE_DIR = "processed_claims"
#     os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)
#     os.makedirs(PROCESSED_BASE_DIR, exist_ok=True)
#     LOGO_PATH = "exl_logo.png"

#     def display_pdf(file_path):
#         with open(file_path, "rb") as f:
#             base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#         st.markdown(pdf_display, unsafe_allow_html=True)



#     # -------------------- DEMAND LETTER EDITOR SCREEN --------------------
#     if "view" in st.session_state and st.session_state["view"] == "demand_package":
#         claim_number = st.session_state["selected_claim"]
#         claim_details = df[df["Claim_Number"] == claim_number].to_dict("records")[0]

#         st.subheader(f"✍️ Edit Demand Letter for Claim {claim_number}")

#         At_fault_party_insurance_carrier_name = "XYZ Insurance Company"
#         Third_party_insurance_carrier_Address = "XYZ Plaza, Columbus, Ohio 43215-2220, USA"
#         Insured_Name = "Reginald Williams"
#         Other_Party_insured_name = "Karen Walton"
#         Date_of_Loss = "09/15/2022"
#         other_party_adjuster_name = "Andrea Duffield"
#         Insured_adjuster_name = "Carol Bradford"

#         # Default demand letter template
#         default_letter = f"""
#         To:
#         {At_fault_party_insurance_carrier_name}
#         {Third_party_insurance_carrier_Address}

#         Re: Subrogation Demand - Claim No. {claim_number}
#         Our Insured: {Insured_Name}
#         Your Insured: {Other_Party_insured_name}
#         Date of Loss: {Date_of_Loss}
#         Loss Location: {claim_details['STATE_GROUP']}

#         Dear {other_party_adjuster_name},

#         We represent {Insured_Name}, the automobile insurance carrier for {Insured_Name}. On {Date_of_Loss},
#         your insured, {Other_Party_insured_name} negligently caused a motor vehicle collision 
#         at {claim_details['STATE_GROUP']}. Based on the police report and supporting evidence, your insured
#         was cited for failure to stop at a red light, thereby establishing liability.

#         As a result of this incident, {Insured_Name} indemnified our insured for the following damages:

#         Category                Amount Paid (USD)
#         ------------------------------------------------
#         Vehicle Repairs             $1260
#         Rental Car Expenses     $500
#         Towing & Storage           $200
#         Medical Payments         $3000
#         ------------------------------------------------
#         Total Demand            $4,960

#         We hereby demand reimbursement in the amount of $4,960 within thirty (30) days
#         of receipt of this letter. Enclosed please find supporting documentation, including
#         proof of payment, repair invoices, photographs, and the police report.

#         Should this matter remain unresolved, we reserve the right to pursue recovery
#         through Arbitration Forums, Inc. or litigation as applicable under state law.

#         Please direct all correspondence and payments to the undersigned.

#         Sincerely,
#         {Insured_adjuster_name}
#         Sr. Relationship Manager
#         ABC Insurance Company
#         Contact: +0001321513312
#         carol@ABCInsurance.com
#             """

#         # Let user edit letter
#         edited_letter = st.text_area("📝 Edit Demand Letter", default_letter, height=600)

#         # Save edited text in session
#         st.session_state["edited_demand_letter"] = edited_letter

#         col1, col2 = st.columns([1,1])
#         with col1:
#             if st.button("⬅️ Back to Subrogation Workbench"):
#                 st.session_state["view"] = "subro_workbench"
#                 st.rerun()


#         with col2:
#             if st.button("✅ Generate Demand Package"):
#                 progress = st.progress(0, text="⚙️ Initializing demand package generation...")

#                 processed_dir = os.path.join(PROCESSED_BASE_DIR, f"{claim_number}")
#                 os.makedirs(processed_dir, exist_ok=True)

#                 OUTPUT_PDF = os.path.join(processed_dir, "Subro_Demand_exhibits_package.pdf")
#                 INTERNAL_PDF = os.path.join(processed_dir, "Internal_adjuster_notes_report.pdf")

#                 # Step 1: Generate demand letter
#                 progress.progress(30, text="📝 Generating demand letter...")
#                 demand_letter_pdf = generate_demand_letter_from_text(edited_letter)
#                 time.sleep(0.5)  # optional for smoother UX

#                 # Step 2: Compile demand package with exhibits
#                 progress.progress(70, text="📑 Compiling exhibits and merging package...")
#                 exhibits = st.session_state["uploaded_docs"][claim_number]
#                 create_demand_package_final_reports(
#                     exhibit_files=exhibits,
#                     output_demand_pdf=OUTPUT_PDF,
#                     claim_id=str(claim_number),
#                     prepared_by="System Auto-Generated",
#                     logo_path=LOGO_PATH,
#                     demand_letter_pdf=demand_letter_pdf  # pass custom edited letter
#                 )
#                 time.sleep(0.5)

#                 # Step 3: Complete
#                 progress.progress(100, text="✅ Demand package generated successfully!")

#                 exhibits_pdf = os.path.join(PROCESSED_BASE_DIR, f"{claim_number}", "Subro_Demand_exhibits_package.pdf")
#                 # Store path in session for preview/download
#                 st.session_state["final_demand_package"] = exhibits_pdf
#                 st.session_state["view"] = "demand_package_preview"

#                 st.success(f"📄 Demand Package generated for Claim {claim_number}")
#                 st.rerun()





#     # -------------------- DEMAND PACKAGE PREVIEW SCREEN --------------------
#     elif "view" in st.session_state and st.session_state["view"] == "demand_package_preview":
#         claim_number = st.session_state["selected_claim"]
#         merged_pdf = st.session_state.get("final_demand_package")

#         st.subheader(f"📦 Final Demand Package Preview - Claim {claim_number}")

#         if merged_pdf and os.path.exists(merged_pdf):
#             # Preview inside Streamlit
#             with open(merged_pdf, "rb") as f:
#                 base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#                 pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
#                 st.markdown(pdf_display, unsafe_allow_html=True)

#             # Download button
#             with open(merged_pdf, "rb") as f:
#                 st.download_button(
#                     label="⬇️ Download Final Demand Package",
#                     data=f,
#                     file_name=f"Final_Demand_Package_{claim_number}.pdf",
#                     mime="application/pdf",
#                     key=f"download_final_{claim_number}"
#                 )

#         if st.button("⬅️ Back to Edit Demand Letter"):
#             st.session_state["view"] = "demand_package"
#             st.rerun()


#     # -------------------- Internal Notes Screen --------------------
#     elif "view" in st.session_state and st.session_state["view"] == "internal_notes":
#         claim_number = st.session_state["selected_claim"]
#         internal_pdf = os.path.join(PROCESSED_BASE_DIR, f"{claim_number}", "Internal_adjuster_notes_report.pdf")

#         st.subheader(f"📝 Internal Adjuster Report for Claim {claim_number}")

#         if os.path.exists(internal_pdf):

#             # Preview PDF
#             display_pdf(internal_pdf)

#             # Download button
#             with open(internal_pdf, "rb") as f:
#                 st.download_button(
#                     label="⬇️ Download Internal Adjuster Report",
#                     data=f,
#                     file_name=f"Internal_adjuster_notes_report_{claim_number}.pdf",
#                     mime="application/pdf",
#                     key=f"download_internal_{claim_number}"
#                 )

#         else:
#             st.warning("⚠️ Internal Adjuster Notes Report not found for this claim.")

#         # Back button
#         if st.button("⬅️ Back to Subrogation Workbench"):
#             st.session_state["view"] = "subro_workbench"
#             st.rerun()

#     # -------------------- Workbench Default View --------------------
#     else:
#         st.title("📑 Subrogation Workbench")

#         actioned_df = df[df["User_Action"].isin(["ASSIGNED"])].copy()

#         if actioned_df.empty:
#             st.info("⚠️ No claims have been assigned to Subrogation Workbench.")
#         else:
#             st.success(f"✅ Showing {len(actioned_df)} claims where actions were saved.")

#             if "uploaded_docs" not in st.session_state:
#                 st.session_state["uploaded_docs"] = {}

#             for idx, row in actioned_df.iterrows():
#                 with st.container(border=True):
#                     col1, col2, col3, col4, col5,col6 = st.columns([1.8,3,1,2,1,1])
#                     with col1:
#                         st.write(f"**Claim #:** {row['Claim_Number']}")
#                     with col2:
#                         selected_action = st.selectbox(
#                             "Action",
#                             [" ","Subrogation Assignment", "A New Witness Added Onto the Claim", "Send First Demand for Subrogation Letter/ Package", "Investigation Pending", "Medical/ Police Report Status", "Close - Not Pursuing"],
#                             key=f"action_{idx}",
#                             index=[" ","Subrogation Assignment", "A New Witness Added Onto the Claim", "Send First Demand for Subrogation Letter/ Package", "Investigation Pending", "Medical/ Police Report Status", "Close - Not Pursuing"].index(row['Subro_User_Action']) if row['Subro_User_Action'] in [" ","Subrogation Assignment", "A New Witness Added Onto the Claim", "Send First Demand for Subrogation Letter/ Package", "Investigation Pending", "Medical/ Police Report Status", "Close - Not Pursuing"] else 0
#                         )

#                     with col3:
#                         if st.button("💾 Save", key=f"save_{idx}"):
#                             df_all = pd.read_csv(data_path)
#                             df_all.at[idx, 'Subro_User_Action'] = selected_action
#                             df_all.to_csv(data_path, index=False)
#                             st.success(f"✅ Action saved for Claim {row['Claim_Number']}")



#                     with col4:
#                         uploaded_files = st.file_uploader(
#                             f"📎 Upload Files (Claim {row['Claim_Number']})",
#                             type=None,
#                             key=f"uploader_{row['Claim_Number']}",
#                             accept_multiple_files=True
#                         )

#                         if uploaded_files:
#                             progress = st.progress(0, text="📂 Initializing upload...")
#                             claim_dir = os.path.join(UPLOAD_BASE_DIR, f"{row['Claim_Number']}")
#                             os.makedirs(claim_dir, exist_ok=True)

#                             # Upload processing with progress
#                             total_steps = len(uploaded_files) + 2  # +2 for reports
#                             step = 0

#                             for i, uploaded_file in enumerate(uploaded_files):
#                                 file_path = os.path.join(claim_dir, uploaded_file.name)
#                                 with open(file_path, "wb") as f:
#                                     f.write(uploaded_file.getbuffer())

#                                 if row['Claim_Number'] not in st.session_state["uploaded_docs"]:
#                                     st.session_state["uploaded_docs"][row['Claim_Number']] = []
#                                 if file_path not in st.session_state["uploaded_docs"][row['Claim_Number']]:
#                                     st.session_state["uploaded_docs"][row['Claim_Number']].append(file_path)

#                                 step += 1
#                                 progress.progress(int((step / total_steps) * 100), 
#                                                 text=f"📂 Uploaded {uploaded_file.name}")

#                                 time.sleep(0.2)  # just to show smooth progress (optional)

#                             # After upload, move to report generation
#                             exhibits = st.session_state["uploaded_docs"][row['Claim_Number']]
#                             processed_dir = os.path.join(PROCESSED_BASE_DIR, f"{row['Claim_Number']}")
#                             os.makedirs(processed_dir, exist_ok=True)

#                             OUTPUT_PDF = os.path.join(processed_dir, "Subro_Demand_exhibits_package.pdf")
#                             INTERNAL_PDF = os.path.join(processed_dir, "Internal_adjuster_notes_report.pdf")
#                             claim_details = df[df["Claim_Number"] == row['Claim_Number']].to_dict("records")[0]

#                             # Step 1: Internal report
#                             step += 1
#                             progress.progress(int((step / total_steps) * 100), text="📝 Generating internal report...")
#                             create_internal_final_reports(
#                                 exhibit_files=exhibits,
#                                 output_internal_pdf=INTERNAL_PDF,
#                                 claim_id=str(claim_details['Claim_Number']),
#                                 prepared_by="System Auto-Generated",
#                                 logo_path=LOGO_PATH
#                             )
#                             time.sleep(0.5)

#                             # Step 2: Demand package (optional)
#                             step += 1
#                             progress.progress(int((step / total_steps) * 100), text="📑 Creating demand package...")
#                             generate_demand_letter_from_text("Auto-generated demand letter")  # Example
#                             time.sleep(0.5)

#                             progress.progress(100, text="✅ Reports generated successfully!")
#                             st.success(f"📄 Reports generated for Claim {row['Claim_Number']}")

#                     # Demand Package button
#                     with col5:
#                         if st.button("📄 Demand Package", key=f"demand_{row['Claim_Number']}"):
#                             st.session_state["selected_claim"] = row["Claim_Number"]
#                             st.session_state["view"] = "demand_package"
#                             st.rerun()

#                     # Internal Notes button
#                     with col6:
#                         if st.button("📝 Internal Report", key=f"notes_{row['Claim_Number']}"):
#                             st.session_state["selected_claim"] = row["Claim_Number"]
#                             st.session_state["view"] = "internal_notes"
#                             st.rerun()
