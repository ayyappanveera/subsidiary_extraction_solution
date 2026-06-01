import requests
import streamlit as st
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Subsidiary Extraction Dashboard",
    layout="wide"
)

st.title("GenAI Subsidiary Extraction Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Upload & Process",
        "Results",
        "Manual Review",
        "Company QA Chatbot",
    ]
)


# -----------------------------
# TAB 1: UPLOAD
# -----------------------------
with tab1:
    st.header("Upload Excel")

    uploaded_file = st.file_uploader(
        "Upload company Excel file",
        type=["xlsx", "xls"]
    )

    if uploaded_file:
        st.success("File selected")

        if st.button("Start Extraction"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state["job_id"] = data["job_id"]

                st.success("Processing started")
                st.code(data["job_id"])
            else:
                st.error(response.text)

    st.subheader("Check Job Status")

    job_id = st.text_input(
        "Job ID",
        value=st.session_state.get("job_id", "")
    )

    if st.button("Refresh Status"):
        if job_id:
            response = requests.get(f"{API_BASE_URL}/job/{job_id}")

            if response.status_code == 200:
                data = response.json()

                st.json(data)

                total = data.get("total_companies", 0)
                completed = data.get("completed_companies", 0)

                if total:
                    st.progress(completed / total)

                if data.get("status") == "completed":
                    st.success("Job completed")

                    download_url = f"{API_BASE_URL}/download/{job_id}"
                    st.link_button("Download Excel Result", download_url)
            else:
                st.error(response.text)


# -----------------------------
# TAB 2: RESULTS
# -----------------------------
with tab2:
    st.header("Extracted Results")

    result_job_id = st.text_input(
        "Enter Job ID for results",
        value=st.session_state.get("job_id", ""),
        key="result_job_id"
    )

    if st.button("Load Results"):
        response = requests.get(f"{API_BASE_URL}/results/{result_job_id}")

        if response.status_code == 200:
            rows = response.json()

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No results found")
        else:
            st.error(response.text)

    st.header("Source Audit")

    if st.button("Load Source Audit"):
        response = requests.get(f"{API_BASE_URL}/sources/{result_job_id}")

        if response.status_code == 200:
            rows = response.json()

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No source audit found")
        else:
            st.error(response.text)


# -----------------------------
# TAB 3: MANUAL REVIEW
# -----------------------------
with tab3:
    st.header("Manual Review / Correction")

    review_job_id = st.text_input(
        "Enter Job ID",
        value=st.session_state.get("job_id", ""),
        key="review_job_id"
    )

    if st.button("Load Pending Results"):
        response = requests.get(f"{API_BASE_URL}/results/{review_job_id}")

        if response.status_code == 200:
            st.session_state["review_rows"] = response.json()
        else:
            st.error(response.text)

    rows = st.session_state.get("review_rows", [])

    if rows:
        selected_id = st.selectbox(
            "Select result ID",
            [r["id"] for r in rows]
        )

        selected = next(r for r in rows if r["id"] == selected_id)

        subsidiary_name = st.text_input(
            "Subsidiary Name",
            value=selected.get("subsidiary_name", "")
        )

        incorporated_location = st.text_input(
            "Incorporated Location",
            value=selected.get("incorporated_location", "")
        )

        holding_percentage = st.text_input(
            "Holding Percentage",
            value=selected.get("holding_percentage", "")
        )

        confidence = st.selectbox(
            "Confidence",
            ["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(
                selected.get("confidence", "Medium")
                if selected.get("confidence", "Medium") in ["High", "Medium", "Low"]
                else "Medium"
            )
        )

        remarks = st.text_area(
            "Remarks",
            value=selected.get("remarks", "")
        )

        review_status = st.selectbox(
            "Review Status",
            ["pending", "approved", "rejected", "corrected"]
        )

        if st.button("Update Review"):
            payload = {
                "subsidiary_name": subsidiary_name,
                "incorporated_location": incorporated_location,
                "holding_percentage": holding_percentage,
                "confidence": confidence,
                "remarks": remarks,
                "review_status": review_status,
            }

            response = requests.put(
                f"{API_BASE_URL}/review/{selected_id}",
                json=payload,
            )

            if response.status_code == 200:
                st.success("Review updated")
            else:
                st.error(response.text)


# -----------------------------
# TAB 4: QA CHATBOT
# -----------------------------
with tab4:
    st.header("Company QA Chatbot")

    st.write(
        "Ask questions about extracted companies, subsidiaries, source URLs, "
        "holding percentage, confidence, and audit data."
    )

    qa_job_id = st.text_input(
        "Job ID for QA",
        value=st.session_state.get("job_id", ""),
        key="qa_job_id"
    )

    company_name = st.text_input(
        "Company Name",
        placeholder="Example: Maersk"
    )

    question = st.text_area(
        "Ask a question",
        placeholder="Example: What subsidiaries were found and what is the holding percentage?"
    )

    if st.button("Ask Chatbot"):
        payload = {
            "job_id": qa_job_id,
            "company_name": company_name,
            "question": question,
        }

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=1200,
        )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data.get("answer", ""))

            if data.get("sources"):
                st.subheader("Sources")
                st.dataframe(
                    pd.DataFrame(data["sources"]),
                    use_container_width=True
                )
        else:
            st.error(response.text)