import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as componentspip
import os
import tempfile
from report_generator import generate_demand_letter_from_text, create_demand_package_final_reports, create_internal_final_reports
import base64
from llm_processing import llm
import shutil
import time
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import numpy as np



# -------------------- Login Credential System --------------------
USER_CREDENTIALS = {
    "Admin": "admin123",
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

# --- Universal CSS ---
st.markdown("""
<style>

/* ======================= */
/* GLOBAL BASE STYLES */
/* ======================= */
header[data-testid="stHeader"], footer {
    visibility: hidden !important;
    height: 0 !important;
}
html, body, [class*="css"] {
    height: 100% !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

/* ======================= */
/* PERMANENT FIXED SIDEBAR */
/* ======================= */

    
    /* --- Hide and disable the sidebar collapse ("<<") button --- */

/* Target the exact div shown in your screenshot */
div[data-testid="stSidebarCollapseButton"] {
    display: none !important;           /* hide the container */
    visibility: hidden !important;
    pointer-events: none !important;
}

/* Additionally, ensure the button inside is disabled even if rendered */
div[data-testid="stSidebarCollapseButton"] button {
    pointer-events: none !important;
    opacity: 0 !important;              /* make invisible */
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* Optional: fix sidebar to stay open */
section[data-testid="stSidebar"] {
    transform: none !important;
    visibility: visible !important;
    width: 280px !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    background-color: #F8F9FA !important;
    border-right: 1px solid #E0E0E0 !important;
    z-index: 1000 !important;
}


/* ======================= */
/* FIXED TOP BANNER */
/* ======================= */
.top-banner {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #185a9d, #0073e6);
    color: white;
    padding: 0 40px 0 320px; /* offset for sidebar width */
    z-index: 900;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

/* Banner text styling */
.banner-center {
    font-size: 26px;
    font-weight: 700;
    text-align: center;
    flex: 1;
}
.banner-right {
    font-size: 16px;
    font-weight: 500;
    text-align: right;
}

/* ======================= */
/* MAIN CONTENT AREA */
/* ======================= */
.block-container {
    position: relative !important;
    padding-top: 90px !important; /* space below banner */
    margin-left: 280px !important; /* permanent sidebar offset */
    width: calc(100vw - 280px) !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

/* ======================= */
/* BUTTON STYLES */
/* ======================= */
.stButton > button {
    background-color: #E3F2FD !important;
    color: #0047AB !important;
    border: 1px solid #E3F2FD !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background-color: #BBDEFB !important;
    color: #003580 !important;
    border-color: #90CAF9 !important;
}



</style>
""", unsafe_allow_html=True)
# --- TOP BANNER HTML ---
st.markdown(f"""
<div class="top-banner">
    <div class="banner-center">
        Claims Litigation360
    </div>
    <div class="banner-right">
        Welcome, {st.session_state.get("username", "User")}
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- Main Content Wrapper --------------------
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

# -------------------- Sidebar --------------------
    with st.sidebar:
        st.image("exl logo.png", use_container_width=True)
        st.markdown("<div style='font-weight:700; font-size:22px;margin-left: 25px; margin-bottom:0px;'>Navigation Panel</div>", unsafe_allow_html=True)
        
        selected_screen = st.radio("", [
            "Attorney Rep Model Referrals",
            "Attorney Rep Reviewed Claims",
            "Legal Propensity Model Referrals",
            "Legal Propensity Reviewed Claims",
            "Monitoring Dashboard",
            "Law Firm Assignment",
            "Legal Spend Dashboard"
        ])

    # -------------------- Load Data --------------------
    # data_path = 'claims_only_Data.csv'
    # data_path = "Syntheticdataset_litigation.csv"
    data_path = "merged_file_1.csv"
    Notes_path = "Notes1.csv"
    att_data_path = "atty_rep_model_predictions.csv"

    @st.cache_data(ttl=0)
    def load_data():
        cdf = pd.read_csv(data_path)
        ndf = pd.read_csv(Notes_path, sep=',')
        df = cdf.merge(ndf[['Claim_Number', 'Claims_Notes',
                       'Summary']], how='left', on='Claim_Number')

        df['ML_SCORE'] = round(df['ML_SCORE'], 2)

        if 'User_Action' not in df.columns:
            df['User_Action'] = ''
        if 'User_Action_Details' not in df.columns:
            df['User_Action_Details'] = ''
        return df
    def att_load_data():
        df = pd.read_csv(att_data_path)
        if 'User_Action' not in df.columns:
            df['User_Action'] = ''
        if 'User_Action_Details' not in df.columns:
            df['User_Action_Details'] = ''

        return df

    df = load_data()
    att_df = att_load_data()
    # df = df[df['Reviewed']==0]
    action_detail_mapping = {
        "Awaiting Additional Info": ["Choose an option", "Field Report", "Inspection Report", "Medical Report", "Police Report", "Property Images", "Pending Demand", "Pending Counter to Offer"],
        "Dismiss": ["Choose an option", "Adequate Reserve", "Already Settled", "Already with Expert", "Already with LL/MCU", "Already with SIU", "Improper Alert", "Arbitration for UM/UIM in this State", "ADR"],
        "Engage an Expert": [],
        "Increase Reserve": [],
        "Refer to Large Loss": [],
        "Refer to SIU": [],
        "Settle": [],
        "Additional Authority Granted": [],
        "In Litigation": ["Transfer Notice to New UM"],
        "In Negotiations": [],
        "Requesting Mediation": []
    }

    att_action_map = {
        "Contact Customer": [
            "Clarify dispute",
            "Set expectations"
        ],
        "Review Coverage": [
            "Explain decision"
        ],
        "Review Handling": [
            "Validate position"
        ],
        "Adjust Strategy": [
            "Request authority",
            "Offer goodwill"
        ],
        "Expedite Claim": [
            "Advance payment",
            "Resolve open items"
        ],
        "Escalate Claim": [
            "Refer to UM",
            "Engage supervisor"
        ],
        "Prepare Defense": [
            "Preserve documentation",
            "Early legal consult"
        ]
    }



    

    # ---------------------Resetting Directories--------------
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


    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- 📊 Dashboard Screen --------------------

if selected_screen == "Attorney Rep Model Referrals":
    st.title("Attorney Rep Claim Referrals")
    # df = load_data()
    df = att_df[att_df['Reviewed'] == 0]
    st.markdown("""
                    <div style='text-align:left; height: 30px; font-size: 25px; margin-top: 10px'>
                        <b>Filter & Search Panel</b>
                    </div>""", unsafe_allow_html=True)
    # --- Create a wrapper DIV for your filter section ---
    st.markdown('<div id="filter-section">', unsafe_allow_html=True)
    filter_cols = st.columns(2)
    st.markdown("""
                    <style>
                    /* Align selectboxes and text_input in a single line */
                    #filter-section div[data-testid="stTextInput"] label,
                    #filter-section div[data-testid="stSelectbox"] label {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight: 600;
                        color: #333;

                    }

                    /* Reduce top margin/padding of Streamlit inputs */
                    #filter-section div[data-testid="stTextInput"], 
                    #filter-section div[data-testid="stSelectbox"] {
                        margin-top: -50px !important;
                    }

                    /* Keep input box heights consistent */
                    #filter-section div[data-testid="stTextInput"] input,
                    #filter-section div[data-testid="stSelectbox"] select {
                        height: 40px !important;
                        padding: 0px 8px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

    with filter_cols[0]:
        peril_filter = st.selectbox(
            "INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

    # with filter_cols[1]:
    #     sub_det = st.selectbox(
    #         "LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')
    with filter_cols[1]:
        claim_search = st.text_input("SEARCH CLAIM NUMBER", key="claim_search")

    # Apply filters
    filtered_df = df.copy()
    if peril_filter != " ":
        filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
    # if sub_det != " ":
    #     filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]

        # filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(
            str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['Model_probabibilty'], ascending=False)
# --- Close the wrapper DIV ---
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.info(
            "⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("Review and Act on Model Recommended Claims")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns(
                [2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8, 4.0, 4.0, 3.0])

            with cols[0]:
                st.markdown(f"**Claim:** {row['CLM_NBR']}")
            with cols[1]:
                st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]:
                st.markdown(f"**Accident State:** {row['ACDNT_ST_ABBR']}")
            with cols[3]:
                st.markdown(
                    f"**Accident City:** {row['ACDNT_CITY']}")
            with cols[4]:
                st.markdown(f"**Clmnt Age** {int(row['DRV_AGE_AT_TIME_OF_LOSS'])}")
            with cols[5]:
                st.markdown(f"**ML Score:** {round(row['Model_probabibilty'], 2)}")

            # --- New Column for Notes Summary Toggle ---
            with cols[6]:
                st.markdown("**Notes**")
                st.markdown("""
                                <style>
                                div[data-testid="stCheckbox"] 
                                {
                                    margin-top: -46px;
                                    margin-left: 6px;
                                    width: 120px !important;
                                    height: 50px !important;
                                    display: flex;
                                }
                                </style>
                            """, unsafe_allow_html=True
                            )

                show_summary = st.toggle("", key=f"notes_toggle_{idx}")

            if show_summary:
                with st.spinner("Generating summary using LLM..."):
                    # Pass the claim notes (or whatever column contains raw notes)
                    summary_text = llm(row['Claim Note'])

                    st.text_area(
                        "Claim Notes Summary",
                        value=summary_text,
                        height=500,
                        key=f"notes_area_{idx}"
                    )

            # --- ACTION COLUMN ---
            with cols[7]:
                # Label above dropdown
                st.markdown(
                    """
                    <div id="action-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                action_options = [
                    "",
                    "Contact Customer",
                    "Review Coverage",
                    "Review Handling",
                    "Adjust Strategy",
                    "Expedite Claim",
                    "Escalate Claim",
                    "Prepare Defense"
                ]

                # Determine default index safely
                user_action_value = row.get("User_Action")
                default_index = action_options.index(
                    user_action_value) if user_action_value in action_options else 0

                # Render Action dropdown
                selected_action = st.selectbox(
                    label="",
                    options=action_options,
                    key=f"action_{idx}",
                    index=default_index,
                    label_visibility="collapsed"
                )

            # --- ACTION DETAILS COLUMN ---
            with cols[8]:
                st.markdown(
                    """
                    <div id="action-details-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action Details</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get detail options based on Action
                if selected_action:
                    action_details_options = att_action_map.get(
                        selected_action, [])
                    options = [] + action_details_options

                    user_action_detail_value = row.get("User_Action_Details")
                    default_index = (
                        options.index(user_action_detail_value)
                        if user_action_detail_value in options
                        else 0
                    )

                    selected_detail = st.selectbox(
                        label="",
                        options=options,
                        key=f"action_details_{idx}",
                        index=default_index,
                        label_visibility="collapsed"
                    )

                else:
                    selected_detail = st.selectbox(
                        label="",
                        options=[""],
                        key=f"action_details_{idx}_disabled",
                        label_visibility="collapsed"
                    )

            # --- UNIVERSAL ALIGNMENT STYLING ---
            st.markdown(
                """
                        <style>
                            /* Consistent layout for both Action and Action Details dropdowns */
                            #action-section div[data-testid="stSelectbox"],
                            #action-details-section div[data-testid="stSelectbox"] {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                                margin-top: -6px !important;   /* Align both dropdowns neatly under label */
                                margin-bottom: 10px;
                                height: auto;
                            }

                            /* Uniform dropdown width and centered text */
                            #action-section div[data-testid="stSelectbox"] select,
                            #action-details-section div[data-testid="stSelectbox"] select {
                                width: 160px;
                                text-align: center;
                            }

                            /* This targets the visible Streamlit selectbox box, version-agnostic */
                            div[data-testid="stSelectbox"] [data-baseweb="select"] > div:nth-child(1)
                        {
                                
                                border-radius: 6px !important;
                                color: #000000
                                /*#185a9d #4682B4!important;*/
                                
                            }
                        </style>
                        """,
                unsafe_allow_html=True
            )
            with cols[9]:
                st.markdown("""
                    <div style= text-align:center; height: 30px;>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)
                # st.markdown("**Save**")
                st.markdown(
                    """
                        <style>
                        div.stButton > button {
                            margin-top: -15px;
                            margin-left: 8px;
                            width: 100px;
                            height: 40px;
                            display: flex;
                            
                        }
                        </style>
                        """,
                    unsafe_allow_html=True
                )
                if st.button("💾", key=f"save_{idx}"):
                    df_all = pd.read_csv(att_data_path)
                    df_all.at[idx, 'User_Action'] = selected_action
                    df_all.at[idx, "User_Action_Details"] = selected_detail
                    df_all.at[idx, "Reviewed"] = 1

                    df_all.to_csv(att_data_path, index=False)

                    st.success(
                        f"✅ Action saved for Claim {row['CLM_NBR']}")

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

elif selected_screen == "Attorney Rep Reviewed Claims":

    st.title("Attorney Rep Reviewed Claims")
    
    df = att_df[att_df['Reviewed'] == 1]
    st.markdown("""
                    <div style='text-align:left; height: 30px; font-size: 25px; margin-top: 10px'>
                        <b>Filter & Search Panel</b>
                    </div>""", unsafe_allow_html=True)
    # --- Create a wrapper DIV for your filter section ---
    st.markdown('<div id="filter-section">', unsafe_allow_html=True)
    filter_cols = st.columns(2)
    st.markdown("""
                    <style>
                    /* Align selectboxes and text_input in a single line */
                    #filter-section div[data-testid="stTextInput"] label,
                    #filter-section div[data-testid="stSelectbox"] label {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight: 600;
                        color: #333;

                    }

                    /* Reduce top margin/padding of Streamlit inputs */
                    #filter-section div[data-testid="stTextInput"], 
                    #filter-section div[data-testid="stSelectbox"] {
                        margin-top: -50px !important;
                    }

                    /* Keep input box heights consistent */
                    #filter-section div[data-testid="stTextInput"] input,
                    #filter-section div[data-testid="stSelectbox"] select {
                        height: 40px !important;
                        padding: 0px 8px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

    with filter_cols[0]:
        peril_filter = st.selectbox(
            "INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

    # with filter_cols[1]:
    #     sub_det = st.selectbox(
    #         "LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')
    with filter_cols[1]:
        claim_search = st.text_input("SEARCH CLAIM NUMBER", key="claim_search")

    # Apply filters
    filtered_df = df.copy()
    if peril_filter != " ":
        filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
    # if sub_det != " ":
    #     filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]

        # filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(
            str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['Model_probabibilty'], ascending=False)

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
        st.info(
            "⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("Reviewed Claims")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns(
                [2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8, 4.0, 4.0, 3.0])

            with cols[0]:
                st.markdown(f"**Claim:** {row['CLM_NBR']}")
            with cols[1]:
                st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]:
                st.markdown(f"**Accident State:** {row['ACDNT_ST_ABBR']}")
            with cols[3]:
                st.markdown(
                    f"**Accident City:** {row['ACDNT_CITY']}")
            with cols[4]:
                st.markdown(f"**Clmnt Age** {int(row['DRV_AGE_AT_TIME_OF_LOSS'])}")
            # with cols[7]: st.markdown(f"**Severity:** {row['CLM_LOSS_SEVERITY_CD']}")
            with cols[5]:
                st.markdown(f"**ML Score:** {round(row['Model_probabibilty'], 2)}")

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
                    summary_text = llm(row['Claim Note'])

                    st.text_area(
                        "Claim Notes Summary",
                        value=summary_text,
                        height=500,
                        key=f"notes_area_{idx}"
                    )

    #         # --- ACTION COLUMN ---
    #         with cols[7]:
    #             # Label above dropdown
    #             st.markdown(
    #                 """
    #                 <div id="action-section" style='text-align:center; margin-bottom:2px;'>
    #                     <b>Action</b>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )

    #             action_options = [
    #                 "",
    #                 "Awaiting Additional Info",
    #                 "Dismiss",
    #                 "Engage an Expert",
    #                 "Increase Reserve",
    #                 "Refer to Large Loss",
    #                 "Refer to SIU",
    #                 "Settle",
    #                 "Additional Authority Granted",
    #                 "In Litigation",
    #                 "In Negotiations",
    #                 "Requesting Mediation"
    #             ]

    #             # Determine default index safely
    #             user_action_value = row.get("User_Action")
    #             default_index = action_options.index(
    #                 user_action_value) if user_action_value in action_options else 0

    #             # Render Action dropdown
    #             selected_action = st.selectbox(
    #                 label="",
    #                 options=action_options,
    #                 key=f"action_{idx}",
    #                 index=default_index,
    #                 label_visibility="collapsed"
    #             )

    #         # --- ACTION DETAILS COLUMN ---
    #         with cols[8]:
    #             st.markdown(
    #                 """
    #                 <div id="action-details-section" style='text-align:center; margin-bottom:2px;'>
    #                     <b>Action Details</b>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )

    #             # Get detail options based on Action
    #             if selected_action:
    #                 action_details_options = action_detail_mapping.get(
    #                     selected_action, [])
    #                 options = [] + action_details_options

    #                 user_action_detail_value = row.get("User_Action_Details")
    #                 default_index = (
    #                     options.index(user_action_detail_value)
    #                     if user_action_detail_value in options
    #                     else 0
    #                 )

    #                 selected_detail = st.selectbox(
    #                     label="",
    #                     options=options,
    #                     key=f"action_details_{idx}",
    #                     index=default_index,
    #                     label_visibility="collapsed"
    #                 )

    #             else:
    #                 selected_detail = st.selectbox(
    #                     label="",
    #                     options=[""],
    #                     key=f"action_details_{idx}_disabled",
    #                     label_visibility="collapsed"
    #                 )

    #         # --- UNIVERSAL ALIGNMENT STYLING ---
    #         st.markdown(
    #             """
    # <style>
    #     /* Consistent layout for both Action and Action Details dropdowns */
    #     #action-section div[data-testid="stSelectbox"],
    #     #action-details-section div[data-testid="stSelectbox"] {
    #         display: flex;
    #         flex-direction: column;
    #         align-items: center;
    #         justify-content: center;
    #         margin-top: -6px !important;   /* Align both dropdowns neatly under label */
    #         margin-bottom: 10px;
    #         height: auto;
    #     }

    #     /* Uniform dropdown width and centered text */
    #     #action-section div[data-testid="stSelectbox"] select,
    #     #action-details-section div[data-testid="stSelectbox"] select {
    #         width: 160px;
    #         text-align: center;
    #     }

    #     /* This targets the visible Streamlit selectbox box, version-agnostic */
    #     div[data-testid="stSelectbox"] [data-baseweb="select"] > div:nth-child(1)
    # {
            
    #         border-radius: 6px !important;
    #         color: #000000
    #         /* #808080 -black #4682B4!important;*/
            
    #     }
    # </style>
    # """,
    #             unsafe_allow_html=True
    #         )
    #         with cols[9]:
    #             st.markdown("""
    #                 <div style= text-align:center; height: 30px;'>
    #                     <b>Save</b>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #             # st.markdown("**Save**")
    #             st.markdown(
    #                 """
    #                     <style>
    #                     div.stButton > button {
    #                         margin-top: -15px;
    #                         margin-left: 8px;
    #                         width: 100px;
    #                         height: 40px;
    #                         display: flex;
    #                     }
    #                     </style>
    #                     """,
    #                 unsafe_allow_html=True
    #             )
    #             if st.button("💾", key=f"save_{idx}"):
    #                 df_all = pd.read_csv(att_data_path)
    #                 df_all.at[idx, 'User_Action'] = selected_action
    #                 df_all.at[idx, "User_Action_Details"] = selected_detail
    #                 df_all.to_csv(att_data_path, index=False)

    #                 st.success(
    #                     f"✅ Action saved for Claim {row['CLM_NBR']}")

                # --- ACTION COLUMN ---
            with cols[7]:
                st.markdown(
                    """
                    <div id="action-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

                # --- Initialize session state ONLY ONCE ---
                action_key = f"action_{idx}"
                if action_key not in st.session_state:
                    saved_action = row.get("User_Action")
                    st.session_state[action_key] = (
                        saved_action if pd.notna(saved_action) else ""
                    )

                selected_action = st.selectbox(
                    label="",
                    options=action_options,
                    key=action_key,
                    label_visibility="collapsed"
                )



            # --- ACTION DETAILS COLUMN ---
            with cols[8]:
                st.markdown(
                    """
                    <div id="action-details-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action Details</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                details_key = f"action_details_{idx}"

                # Get saved detail safely
                saved_detail = row.get("User_Action_Details")
                saved_detail = saved_detail if pd.notna(saved_detail) else ""

                if selected_action:
                    action_details_options = action_detail_mapping.get(selected_action, [])

                    # 👉 Always include saved value (prevents silent reset)
                    options = [""] + list(dict.fromkeys(
                        action_details_options + ([saved_detail] if saved_detail else [])
                    ))

                    # 👉 Initialize ONLY ONCE
                    if details_key not in st.session_state:
                        st.session_state[details_key] = saved_detail

                    selected_detail = st.selectbox(
                        label="",
                        options=options,
                        key=details_key,
                        label_visibility="collapsed"
                    )

                else:
                    st.session_state[details_key] = ""
                    selected_detail = st.selectbox(
                        label="",
                        options=[""],
                        key=f"{details_key}_disabled",
                        label_visibility="collapsed"
                    )


            # --- SAVE BUTTON ---
            with cols[9]:
                st.markdown("""
                    <div style='text-align:center; height: 30px;'>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    """
                    <style>
                    div.stButton > button {
                        margin-top: -15px;
                        margin-left: 8px;
                        width: 100px;
                        height: 40px;
                        display: flex;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("💾", key=f"save_{idx}"):
                    df_all = pd.read_csv(att_data_path)

                    df_all.at[idx, "User_Action"] = st.session_state[action_key]
                    df_all.at[idx, "User_Action_Details"] = st.session_state[details_key]

                    df_all.to_csv(att_data_path, index=False)

                    st.success(f"✅ Action saved for Claim {row['CLM_NBR']}")

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





# -------------------- 📊 Dashboard Screen --------------------

elif selected_screen == "Legal Propensity Model Referrals":
    st.title("Legal Propensity Model Referrals")
    df = load_data()
    df = df[df['Reviewed'] == 0]
    st.markdown("""
                    <div style='text-align:left; height: 30px; font-size: 25px; margin-top: 10px'>
                        <b>Filter & Search Panel</b>
                    </div>""", unsafe_allow_html=True)
    # --- Create a wrapper DIV for your filter section ---
    st.markdown('<div id="filter-section">', unsafe_allow_html=True)
    filter_cols = st.columns(3)
    st.markdown("""
                    <style>
                    /* Align selectboxes and text_input in a single line */
                    #filter-section div[data-testid="stTextInput"] label,
                    #filter-section div[data-testid="stSelectbox"] label {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight: 600;
                        color: #333;

                    }

                    /* Reduce top margin/padding of Streamlit inputs */
                    #filter-section div[data-testid="stTextInput"], 
                    #filter-section div[data-testid="stSelectbox"] {
                        margin-top: -50px !important;
                    }

                    /* Keep input box heights consistent */
                    #filter-section div[data-testid="stTextInput"] input,
                    #filter-section div[data-testid="stSelectbox"] select {
                        height: 40px !important;
                        padding: 0px 8px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

    with filter_cols[0]:
        peril_filter = st.selectbox(
            "INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

    with filter_cols[1]:
        sub_det = st.selectbox(
            "LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')
    with filter_cols[2]:
        claim_search = st.text_input("SEARCH CLAIM NUMBER", key="claim_search")

    # Apply filters
    filtered_df = df.copy()
    if peril_filter != " ":
        filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
    if sub_det != " ":
        filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]

        # filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(
            str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['ML_SCORE'], ascending=False)
# --- Close the wrapper DIV ---
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.info(
            "⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("Review and Act on Model Recommended Claims")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns(
                [2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8, 4.0, 4.0, 3.0])

            with cols[0]:
                st.markdown(f"**Claim:** {row['Claim_Number']}")
            with cols[1]:
                st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]:
                st.markdown(f"**State:** {row['FTR_JRSDTN_ST_ABBR']}")
            with cols[3]:
                st.markdown(
                    f"**Body Part Injured:** {row['BODY_PART_INJD_DESC']}")
            with cols[4]:
                st.markdown(f"**Loss Party:** {row['LOSS_PARTY']}")
            with cols[5]:
                st.markdown(f"**ML Score:** {row['ML_SCORE']}")

            # --- New Column for Notes Summary Toggle ---
            with cols[6]:
                st.markdown("**Notes**")
                st.markdown("""
                                <style>
                                div[data-testid="stCheckbox"] 
                                {
                                    margin-top: -46px;
                                    margin-left: 6px;
                                    width: 120px !important;
                                    height: 50px !important;
                                    display: flex;
                                }
                                </style>
                            """, unsafe_allow_html=True
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

            # --- ACTION COLUMN ---
            with cols[7]:
                # Label above dropdown
                st.markdown(
                    """
                    <div id="action-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

                # Determine default index safely
                user_action_value = row.get("User_Action")
                default_index = action_options.index(
                    user_action_value) if user_action_value in action_options else 0

                # Render Action dropdown
                selected_action = st.selectbox(
                    label="",
                    options=action_options,
                    key=f"action_{idx}",
                    index=default_index,
                    label_visibility="collapsed"
                )

            # --- ACTION DETAILS COLUMN ---
            with cols[8]:
                st.markdown(
                    """
                    <div id="action-details-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action Details</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get detail options based on Action
                if selected_action:
                    action_details_options = action_detail_mapping.get(
                        selected_action, [])
                    options = [] + action_details_options

                    user_action_detail_value = row.get("User_Action_Details")
                    default_index = (
                        options.index(user_action_detail_value)
                        if user_action_detail_value in options
                        else 0
                    )

                    selected_detail = st.selectbox(
                        label="",
                        options=options,
                        key=f"action_details_{idx}",
                        index=default_index,
                        label_visibility="collapsed"
                    )

                else:
                    selected_detail = st.selectbox(
                        label="",
                        options=[""],
                        key=f"action_details_{idx}_disabled",
                        label_visibility="collapsed"
                    )

            # --- UNIVERSAL ALIGNMENT STYLING ---
            st.markdown(
                """
                        <style>
                            /* Consistent layout for both Action and Action Details dropdowns */
                            #action-section div[data-testid="stSelectbox"],
                            #action-details-section div[data-testid="stSelectbox"] {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                                margin-top: -6px !important;   /* Align both dropdowns neatly under label */
                                margin-bottom: 10px;
                                height: auto;
                            }

                            /* Uniform dropdown width and centered text */
                            #action-section div[data-testid="stSelectbox"] select,
                            #action-details-section div[data-testid="stSelectbox"] select {
                                width: 160px;
                                text-align: center;
                            }

                            /* This targets the visible Streamlit selectbox box, version-agnostic */
                            div[data-testid="stSelectbox"] [data-baseweb="select"] > div:nth-child(1)
                        {
                                
                                border-radius: 6px !important;
                                color: #000000
                                /*#185a9d #4682B4!important;*/
                                
                            }
                        </style>
                        """,
                unsafe_allow_html=True
            )
            with cols[9]:
                st.markdown("""
                    <div style= text-align:center; height: 30px;>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)
                # st.markdown("**Save**")
                st.markdown(
                    """
                        <style>
                        div.stButton > button {
                            margin-top: -15px;
                            margin-left: 8px;
                            width: 100px;
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
                    df_all.at[idx, "Reviewed"] = 1

                    df_all.to_csv(data_path, index=False)

                    st.success(
                        f"✅ Action saved for Claim {row['Claim_Number']}")

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

elif selected_screen == "Legal Propensity Reviewed Claims":

    st.title("Legal Propensity Reviewed Claims")
    df = load_data()
    df = df[df['Reviewed'] == 1]
    st.markdown("""
                    <div style='text-align:left; height: 30px; font-size: 25px; margin-top: 10px'>
                        <b>Filter & Search Panel</b>
                    </div>""", unsafe_allow_html=True)
    # --- Create a wrapper DIV for your filter section ---
    st.markdown('<div id="filter-section">', unsafe_allow_html=True)
    filter_cols = st.columns(3)
    st.markdown("""
                    <style>
                    /* Align selectboxes and text_input in a single line */
                    #filter-section div[data-testid="stTextInput"] label,
                    #filter-section div[data-testid="stSelectbox"] label {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight: 600;
                        color: #333;

                    }

                    /* Reduce top margin/padding of Streamlit inputs */
                    #filter-section div[data-testid="stTextInput"], 
                    #filter-section div[data-testid="stSelectbox"] {
                        margin-top: -50px !important;
                    }

                    /* Keep input box heights consistent */
                    #filter-section div[data-testid="stTextInput"] input,
                    #filter-section div[data-testid="stSelectbox"] select {
                        height: 40px !important;
                        padding: 0px 8px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

    with filter_cols[0]:
        peril_filter = st.selectbox(
            "INCIDENT CAUSE", [" "] + list(df['COL_CD'].unique()), key='peril_filter')

    with filter_cols[1]:
        sub_det = st.selectbox(
            "LOB SUB-LOB", [" "] + list(df['SUB_DTL_DESC'].unique()), key='sub_det_filter')
    with filter_cols[2]:
        claim_search = st.text_input("SEARCH CLAIM NUMBER", key="claim_search")

    # Apply filters
    filtered_df = df.copy()
    if peril_filter != " ":
        filtered_df = filtered_df[filtered_df['COL_CD'] == peril_filter]
    if sub_det != " ":
        filtered_df = filtered_df[filtered_df['SUB_DTL_DESC'] == sub_det]

        # filtered_df = df.copy()

    # Apply claim number search if entered
    if claim_search.strip():
        filtered_df = filtered_df[filtered_df['Claim_Number'].astype(
            str).str.contains(claim_search.strip(), case=False)]

    # suspicious_df = filtered_df[filtered_df['Prediction'] == 1].copy()
    suspicious_df = filtered_df.sort_values(by=['ML_SCORE'], ascending=False)

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
        st.info(
            "⚠️ No suspected litigated claims found with current filters or search.")
    else:
        st.subheader("Reviewed Claims")

        for idx, row in suspicious_df.iterrows():
            st.markdown("---")
            cols = st.columns(
                [2.2, 3.8, 1.8, 4.2, 3.3, 2.2, 1.8, 4.0, 4.0, 3.0])

            with cols[0]:
                st.markdown(f"**Claim:** {row['Claim_Number']}")
            with cols[1]:
                st.markdown(f"**Incident Cause:** {row['COL_CD']}")
            with cols[2]:
                st.markdown(f"**State:** {row['FTR_JRSDTN_ST_ABBR']}")
            # with cols[3]: st.markdown(f"**Paid:** ${row['PAID_FINAL']:.2f}")
            # with cols[3]: st.markdown(f"**Accident City:** {row['ACDNT_CITY']}")
            with cols[3]:
                st.markdown(
                    f"**Body Part Injured:** {row['BODY_PART_INJD_DESC']}")
            with cols[4]:
                st.markdown(f"**Loss Party:** {row['LOSS_PARTY']}")
            # with cols[7]: st.markdown(f"**Severity:** {row['CLM_LOSS_SEVERITY_CD']}")
            with cols[5]:
                st.markdown(f"**ML Score:** {row['ML_SCORE']}")

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

            # --- ACTION COLUMN ---
            with cols[7]:
                # Label above dropdown
                st.markdown(
                    """
                    <div id="action-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

                # Determine default index safely
                user_action_value = row.get("User_Action")
                default_index = action_options.index(
                    user_action_value) if user_action_value in action_options else 0

                # Render Action dropdown
                selected_action = st.selectbox(
                    label="",
                    options=action_options,
                    key=f"action_{idx}",
                    index=default_index,
                    label_visibility="collapsed"
                )

            # --- ACTION DETAILS COLUMN ---
            with cols[8]:
                st.markdown(
                    """
                    <div id="action-details-section" style='text-align:center; margin-bottom:2px;'>
                        <b>Action Details</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get detail options based on Action
                if selected_action:
                    action_details_options = action_detail_mapping.get(
                        selected_action, [])
                    options = [] + action_details_options

                    user_action_detail_value = row.get("User_Action_Details")
                    default_index = (
                        options.index(user_action_detail_value)
                        if user_action_detail_value in options
                        else 0
                    )

                    selected_detail = st.selectbox(
                        label="",
                        options=options,
                        key=f"action_details_{idx}",
                        index=default_index,
                        label_visibility="collapsed"
                    )

                else:
                    selected_detail = st.selectbox(
                        label="",
                        options=[""],
                        key=f"action_details_{idx}_disabled",
                        label_visibility="collapsed"
                    )

            # --- UNIVERSAL ALIGNMENT STYLING ---
            st.markdown(
                """
    <style>
        /* Consistent layout for both Action and Action Details dropdowns */
        #action-section div[data-testid="stSelectbox"],
        #action-details-section div[data-testid="stSelectbox"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: -6px !important;   /* Align both dropdowns neatly under label */
            margin-bottom: 10px;
            height: auto;
        }

        /* Uniform dropdown width and centered text */
        #action-section div[data-testid="stSelectbox"] select,
        #action-details-section div[data-testid="stSelectbox"] select {
            width: 160px;
            text-align: center;
        }

        /* This targets the visible Streamlit selectbox box, version-agnostic */
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div:nth-child(1)
    {
            
            border-radius: 6px !important;
            color: #000000
            /* #808080 -black #4682B4!important;*/
            
        }
    </style>
    """,
                unsafe_allow_html=True
            )
            with cols[9]:
                st.markdown("""
                    <div style= text-align:center; height: 30px;'>
                        <b>Save</b>
                    </div>
                    """, unsafe_allow_html=True)
                # st.markdown("**Save**")
                st.markdown(
                    """
                        <style>
                        div.stButton > button {
                            margin-top: -15px;
                            margin-left: 8px;
                            width: 100px;
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

                    st.success(
                        f"✅ Action saved for Claim {row['Claim_Number']}")

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


# -------------------- 📊 Monitoring Dashboard --------------------
elif selected_screen == "Monitoring Dashboard":
    # st.title("Monitoring Dashboard - Power BI")

    # st.markdown("#### Embedded Power BI Dashboard Below:")

    powerbi_embed_url = """
<iframe title="Litigation Propensity Model Monitoring Dashboard" width="1200" height="800" src="https://app.powerbi.com/reportEmbed?reportId=05566686-41f7-46ff-a15d-df70a00f4f3f&autoAuth=true&ctid=dafe49bc-5ac3-4310-97b4-3e44a28cbf18" frameborder="0" allowFullScreen="true"></iframe>  
    """
    componentspip.html(powerbi_embed_url, height=800)


elif selected_screen == "Law Firm Assignment":
    st.title("Law Firm Assignment")

    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------
    # Page Config
    # -------------------------
    st.set_page_config("Law Firm Assignment", layout="wide")



    @st.cache_data
    def load_data():
        claims = pd.read_csv("model_output_with_predicted_cluster.csv")
        firms = pd.read_csv("synthetic_litigation_dataset_with_firms_and_cluster.csv")
        return claims, firms

    claims_df, firms_df = load_data()



    # -------------------------
    # Custom CSS
    # -------------------------
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .title-card {
        background-color: #1f3c88;
        padding: 14px;
        border-radius: 8px;
        color: white;
    }
    .small-text {
        font-size: 13px;
        color: #555;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        background-color: #e6f0ff;
        display: inline-block;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------
    # Claim Context Bar
    # -------------------------
    c1, c2, c3, c4, c5 = st.columns([1,0.5,4,1,1])

    with c1: 
            # -------------------------
        # Claim Selection
        # -------------------------
        claim_idx = st.selectbox(
            "Select Claim",
            claims_df.index,
            format_func=lambda x: f"Claim {x}"
        )

        claim = claims_df.loc[claim_idx]
        # st.selectbox("Claim Context", ["CLM-10245", "CLM-10246"])

    with c2:
        st.metric("Jurisdiction", claim["FTR_JRSDTN_ST_ABBR"])
        # st.markdown("**State**")
        # st.write("TX")

    with c3:
        st.metric("Injury Description", claim["Short_BODY_PART_INJD_DESC"])
        
    with c4:
        st.metric("Demand", claim["DEMAND"])

        # st.markdown("**Exposure**")
        # st.write("$250K - $500K")

    with c5:
        st.metric("Offer", claim["OFFER"])
        # st.markdown("**Stage**")
        # st.write("OFFER")

    st.divider()

    # -------------------------
    # Main Section
    # -------------------------
    left, right = st.columns([2.5, 1.5])

        
    with left:
        st.subheader("Legal Market Segment Identification")

        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import numpy as np

        # -----------------------------
        # Load data
        # -----------------------------
        df = pd.read_csv("firm_level_cluster_map_output.csv")

        # Separate firms and centroids
        firms = df[df["entity_type"] == "firm"]
        centroids = df[df["entity_type"] == "centroid"]

        # -----------------------------
        # Cluster color mapping
        # -----------------------------
        cluster_colors = {
            "Efficient Volume Handlers": "#F4B400",      # yellow
            "Outcome Specialists": "#6A5ACD",            # purple
            "High-Value Core Firms": "#66C2A5",           # green
            "High-Cost/ Underperformers": "#5DA5DA"      # blue
        }

        # -----------------------------
        # Create plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(12, 7))

        # -----------------------------
        # Scatter points (firms)
        # -----------------------------
        for cluster, color in cluster_colors.items():
            subset = firms[firms["cluster_id"] == cluster]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                s=50,
                alpha=0.8,
                color=color,
                label=cluster
                    )

        # -----------------------------
        for cluster, color in cluster_colors.items():
            cluster_points = firms[firms["cluster_id"] == cluster][["PC1", "PC2"]]

            # Convex hull requires at least 3 points
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points.values)
                hull_points = cluster_points.values[hull.vertices]

                polygon = Polygon(
                    hull_points,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.5
                )

                ax.add_patch(polygon)



        # -----------------------------
        ax.scatter(
            centroids["PC1"],
            centroids["PC2"],
            color="black",
            s=120,
            marker="X",
            label="Cluster Centroid"
        )

    # -----------------------------
    # Highlight "This Claim" (dynamic firm)
    # -----------------------------
        # a = "Phillips & Garcia, LLP"
        a = claim["Cluster_name"]

        this_claim = firms[firms["firm_name"] == claim["Firm Name"]]

        if not this_claim.empty:
            x = this_claim.iloc[0]["PC1"]
            y = this_claim.iloc[0]["PC2"]

            ax.scatter(
                x,
                y,
                s=300,
                facecolor="white",
                edgecolor="black",
                linewidth=2.5,
                zorder=6
            )

            ax.text(
                x,
                y + 0.35,
                "This Claim",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="#4B0082",
                    ec="none",
                    alpha=0.9
                ),
                color="white",
                zorder=7
            )
        else:
            st.warning(f"Firm '{a}' not found in data.")

        # -----------------------------
        # Labels & styling
        # -----------------------------
        # ax.set_title("Legal Market Segment Identification\nClaim Clustering Map", fontsize=14)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.axhline(0, linestyle="--", alpha=0.3)
        ax.axvline(0, linestyle="--", alpha=0.3)

        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        # plt.show()
        st.pyplot(fig)








    # -------------------------
    # Assigned Firm Cluster
    # -------------------------
    with right:
        # a = "Phillips & Garcia, LLP"

        # Lit_data_firm_df = pd.read_csv("synthetic_litigation_dataset_with_firms_and_cluster.csv")
        # Cluster_id = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == a]["Cluster_name"].values[0]
        # Avg_Cycle_Time = int(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Cycle time"].mean())
        # Win_Rate = round(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Win rate proxy"].mean()*100,1)
        # Avg_Cost = int(Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Cost per case"].mean())
        # Avg_claim_closed_cnt = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Case closed count"]
        # Avg_claim_closed_cnt = int((1 / (1 + np.exp(-Avg_claim_closed_cnt.mean()))) * 100)
        # Paid_post_appeal = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]["Paid post appeal"]
        # Avg_paid_post_appeal = int((1 / (1 + np.exp(-Paid_post_appeal.mean()))) * 100)




        # cluster_profile_map = {
        #     "High-Value Core Firms": [
        #         "Strong win rates with balanced cost and cycle time",
        #         "Premium fee structure",
        #         "Reliable for high-value cases"
        #     ],

        #     "High-Cost / Underperformers": [
        #         "High cost per case",
        #         "Weaker win rates and slower cycle times",
        #         "Requires tighter performance governance"
        #     ],

        #     "Outcome Specialists": [
        #         "Highest win rates",
        #         "Moderate case volumes",
        #         "Best suited for complex, outcome-critical matters"
        #     ],

        #     "Efficient Volume Handlers": [
        #         "Low cost per case",
        #         "Fast cycle times",
        #         "Ideal for high-volume, routine work"
        #     ]
        # }


        # cluster_profile = cluster_profile_map.get(Cluster_id, ["No profile available."])



        # st.markdown(f"""
        # <div class="metric-card">
        #     <h4>Assigned Firm Cluster</h4>
        #     <h5>{Cluster_id}</h5>
        #     <hr>
        #     <b>Avg Cycle Time:</b> {Avg_Cycle_Time * 100} days<br>
        #     <b>Win Rate:</b> {Win_Rate}%<br>
        #     <b>Avg Cost:</b> ${Avg_Cost *10}K<br>
        #     <b>Avg Claim Closed Count:</b> {Avg_claim_closed_cnt}<br>
        #     <b>Avg Paid Post Appeal:</b> ${Avg_paid_post_appeal}K<br>

        #     <hr>
        #     <b>Typical Claim Profile</b>
        #     <ul>
        #         <li>{cluster_profile[0]}</li>
        #         <li>{cluster_profile[1]}</li>
        #         <li>{cluster_profile[2]}</li>
        #     </ul>
        #     <span class="small-text">22
        #     Best suited for long-duration, high-exposure litigation.
        #     </span>

        # </div>
        # """, unsafe_allow_html=True)


        Lit_data_firm_df = pd.read_csv(
            "synthetic_litigation_dataset_with_firms_and_cluster.csv"
        )

        selected_firm = claim["Firm Name"]

        firm_row = Lit_data_firm_df[
            Lit_data_firm_df["Firm Name"] == selected_firm
        ]

        if firm_row.empty:
            st.error(f"Firm '{selected_firm}' not found in dataset.")
        else:
            Cluster_id = firm_row["Cluster_name"].iloc[0]

            cluster_df = Lit_data_firm_df[
                Lit_data_firm_df["Cluster_name"] == Cluster_id
            ]

            Avg_Cycle_Time = int(cluster_df["Cycle time"].mean())
            Win_Rate = round(cluster_df["Win rate proxy"].mean() * 100, 1)
            Avg_Cost = int(cluster_df["Cost per case"].mean())

            Avg_claim_closed_cnt = int(
                (1 / (1 + np.exp(-cluster_df["Case closed count"].mean()))) * 100
            )

            Avg_paid_post_appeal = int(
                (1 / (1 + np.exp(-cluster_df["Paid post appeal"].mean()))) * 100
            )

            cluster_profile_map = {
                "High-Value Core Firms": [
                    "Strong win rates with balanced cost and cycle time",
                    "Premium fee structure",
                    "Reliable for high-value cases"
                ],
                "High-Cost / Underperformers": [
                    "High cost per case",
                    "Weaker win rates and slower cycle times",
                    "Requires tighter performance governance"
                ],
                "Outcome Specialists": [
                    "Highest win rates",
                    "Moderate case volumes",
                    "Best suited for complex, outcome-critical matters"
                ],
                "Efficient Volume Handlers": [
                    "Low cost per case",
                    "Fast cycle times",
                    "Ideal for high-volume, routine work"
                ]
            }

            cluster_profile = cluster_profile_map.get(
                Cluster_id, ["No profile available."] * 3
            )

            st.markdown(f"""
            <div class="metric-card">
                <h4>Assigned Firm Cluster</h4>
                <h5>{Cluster_id}</h5>
                <hr>
                <b>Avg Cycle Time:</b> {Avg_Cycle_Time} days<br>
                <b>Win Rate:</b> {Win_Rate}%<br>
                <b>Avg Cost:</b> ${Avg_Cost:,}<br>
                <b>Avg Claim Closed Count:</b> {Avg_claim_closed_cnt}<br>
                <b>Avg Paid Post Appeal:</b> ${Avg_paid_post_appeal}K<br>
                <hr>
                <b>Typical Claim Profile</b>
                <ul>
                    <li>{cluster_profile[0]}</li>
                    <li>{cluster_profile[1]}</li>
                    <li>{cluster_profile[2]}</li>
                </ul> 
                <span class="small-text">
                Best suited for long-duration, high-exposure litigation. 
                </span>
            </div>
            """, unsafe_allow_html=True)


    st.divider()

    # -------------------------
    # Recommended Firms
    # -------------------------

    st.subheader("Optimization Strategy")

    strategy = st.radio(
        "Select Recommendation Strategy",
        ["Outcome Focused", "Cost Focused"],
        index=0,
        horizontal=True
    )

    if strategy == "Outcome Focused":


        st.subheader("Recommended Firms (Within Cluster)")
        col1, col2, col3 = st.columns(3)
        with col1:
            cycle_time_weight = st.slider(
                "Cycle Time Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        with col2:
            win_rate_weight = st.slider(
                "Win Rate Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        with col3:
            Cost_per_case_weight = st.slider(
                "Cost Per Case Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )




        Lit_data_firm_filter_df = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]



        state = claim["FTR_JRSDTN_ST_ABBR"]


        Lit_data_firm_filter_df = Lit_data_firm_filter_df[
            Lit_data_firm_filter_df["state_list"].str.contains(
                fr"\b{state}\b", na=False
            )
        ]


        # ---- Compute weighted score ----
        Lit_data_firm_filter_df['Weighted_Score'] = (
            cycle_time_weight * (10 - Lit_data_firm_filter_df['Cycle time']) +
            win_rate_weight * Lit_data_firm_filter_df['Win rate proxy'] +
            Cost_per_case_weight * (10 - Lit_data_firm_filter_df['Cost per case'])
        )

        top_firms = Lit_data_firm_filter_df.sort_values(
            "Weighted_Score", ascending=False
        ).head(3)

        # ---- Layout ----
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])

        # ---- Build firms list SAFELY ----
        firms = []


        for _, row in top_firms.iterrows():
            firms.append(
                (
                    row["Firm Name"],
                    row["state_list"],
                    f"{round(row['Win rate proxy'] * 100, 1)}%",
                    f"${int(row['Cost per case']*10)}K",
                    f"{int(row['Cycle time']) *100} Days",
                    row['Firm_Profile']
                )
            )

        # ---- Guard: no firms found ----
        if not firms:
            st.warning("No recommended firms found for the selected criteria.")
        else:
            # ---- Render firm cards ----
            for col, firm in zip([f1, f2, f3], firms):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h5>{firm[0]} ✅</h5>
                            <span class = "large-text">{state}</span>
                            <hr>
                            <b>Win Rate:</b> {firm[2]}<br>
                            <b>Avg Cost:</b> {firm[3]}<br>
                            <b>Cycle Time:</b> {firm[4]}<br>
                            <b>Firm Profile:</b> {firm[5]}<br>

                            
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        # -------------------------
        # Decision Support Summary
        # -------------------------
        with f4:
            progress = claim["proba"]
            progres = round(claim["proba"] * 100, 1)

            st.markdown(f"""
            <div class="metric-card">
                <h5>Decision Support Summary</h5>
                <b>Model Confidence {progres}%</b>
            </div>
            """, unsafe_allow_html=True)

    # e.g., 0.84 for 84%

            st.progress(progress)

            st.markdown("""
            <span class="small-text">
            Decision support only.<br>
            Final assignment at adjuster discretion.
            </span>
            """, unsafe_allow_html=True)

    else:
        st.subheader("Recommended Firms (Within Cluster)")
        col1, col2 = st.columns(2)
        with col1:
            cost_per_case = st.slider(
                "Cost Per Case Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        with col2:
            paid_post_appeal = st.slider(
                "Paid Post Appeal Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )




        Lit_data_firm_filter_df = Lit_data_firm_df[Lit_data_firm_df["Cluster_name"] == Cluster_id]



        state = claim["FTR_JRSDTN_ST_ABBR"]


        Lit_data_firm_filter_df = Lit_data_firm_filter_df[
            Lit_data_firm_filter_df["state_list"].str.contains(
                fr"\b{state}\b", na=False
            )
        ]


        # ---- Compute weighted score ----
        Lit_data_firm_filter_df['Weighted_Score'] = (
            cost_per_case * (Lit_data_firm_filter_df['Cost per case']) +
            paid_post_appeal * Lit_data_firm_filter_df['Paid post appeal'] 
            )

        top_firms = Lit_data_firm_filter_df.sort_values(
            "Weighted_Score", ascending=False
        ).head(3)

        # ---- Layout ----
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])

        # ---- Build firms list SAFELY ----
        firms = []


        for _, row in top_firms.iterrows():
            firms.append(
                (
                    row["Firm Name"],
                    row["state_list"],
                    f"{round(row['Win rate proxy'] * 100, 1)}%",
                    f"${int(row['Cost per case']*10)}K",
                    f"{int(row['Cycle time']) *100} Days",
                    row['Firm_Profile']
                )
            )

        # ---- Guard: no firms found ----
        if not firms:
            st.warning("No recommended firms found for the selected criteria.")
        else:
            # ---- Render firm cards ----
            for col, firm in zip([f1, f2, f3], firms):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h5>{firm[0]} ✅</h5>
                            <span class = "large-text">{state}</span>
                            <hr>
                            <b>Win Rate:</b> {firm[2]}<br>
                            <b>Avg Cost:</b> {firm[3]}<br>
                            <b>Cycle Time:</b> {firm[4]}<br>
                            <b>Firm Profile:</b> {firm[5]}<br>

                            
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        # -------------------------
        # Decision Support Summary
        # -------------------------
        with f4:
            progress = claim["proba"]
            progres = round(claim["proba"] * 100, 1)

            st.markdown(f"""
            <div class="metric-card">
                <h5>Decision Support Summary</h5>
                <b>Model Confidence {progres}%</b>
            </div>
            """, unsafe_allow_html=True)

    # e.g., 0.84 for 84%

            st.progress(progress)

            st.markdown("""
            <span class="small-text">
            Decision support only.<br>
            Final assignment at adjuster discretion.
            </span>
            """, unsafe_allow_html=True)




# -------------------- 📊 Monitoring Dashboard --------------------
elif selected_screen == "Legal Spend Dashboard":
    # st.title("Legal Spend Dashboard - Power BI")

    # st.markdown("#### Embedded Power BI Dashboard Below:")

    powerbi_embed_url = """
 <iframe title="Legal Expense Dashboard" width="1200" height="800" src=https://app.powerbi.com/reportEmbed?reportId=dbc06794-896c-494e-a901-caa07670f375&autoAuth=true&ctid=dafe49bc-5ac3-4310-97b4-3e44a28cbf18
 frameborder="0" allowFullScreen="true"></iframe>

    """
    componentspip.html(powerbi_embed_url, height=800)


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

