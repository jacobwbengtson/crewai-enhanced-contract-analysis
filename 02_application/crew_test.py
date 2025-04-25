# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai import LLM
import os
from datetime import datetime

st.set_page_config(page_title="CrewAI Connectivity Test", page_icon="🧪")

st.title("CrewAI Connectivity Test")
st.write("This app tests if CrewAI can connect to OpenAI API successfully")

# API Key input
api_key = st.text_input("OpenAI API Key", type="password")

# Model selection
model_options = [
    "openai/gpt-4",
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229"
]
selected_model = st.selectbox("Select Model", model_options)

# Message
test_message = st.text_input("Test message for agent",
                             value="Explain what makes a good legal contract in one paragraph.")

# Test button
if st.button("Test CrewAI Connection"):
    if not api_key:
        st.error("Please enter your API key")
    else:
        try:
            # Set the API key
            os.environ["OPENAI_API_KEY"] = api_key

            start_time = datetime.now()
            st.info(f"Test started at {start_time.strftime('%H:%M:%S')}")

            with st.spinner("Testing CrewAI connection..."):
                # Create LLM
                llm = LLM(
                    model=selected_model,
                    temperature=0.7,
                    max_tokens=150
                )

                # Create a test agent
                test_agent = Agent(
                    role="Legal Advisor",
                    goal="Provide clear, concise legal advice",
                    backstory="You are an experienced legal professional with expertise in contract law.",
                    llm=llm,
                    verbose=True
                )

                # Create a simple task WITH expected_output parameter
                test_task = Task(
                    description=test_message,
                    expected_output="A concise paragraph about legal contracts",
                    agent=test_agent
                )

                # Create and run the crew
                crew = Crew(
                    agents=[test_agent],
                    tasks=[test_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Execute
                result = crew.kickoff()

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                st.success(f"✅ Test completed in {duration:.2f} seconds!")

                # Output the result (convert to string for display)
                st.subheader("Response:")
                result_str = str(result)  # Convert the CrewOutput to string
                st.write(result_str)

                # Display additional debug info
                with st.expander("Debug Information"):
                    st.write(f"- Model: {selected_model}")
                    st.write(f"- Start time: {start_time}")
                    st.write(f"- End time: {end_time}")
                    st.write(f"- Duration: {duration:.2f} seconds")
                    st.write(f"- Result type: {type(result).__name__}")
                    # Don't try to get length of result object
                    if hasattr(result, "__str__"):
                        st.write(f"- Result string length: {len(str(result))} characters")

                # Show the raw result object
                with st.expander("Raw Result Object"):
                    st.write(result)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Check the traceback below for details:")
            import traceback

            st.code(traceback.format_exc())

# Add a section explaining what's happening
with st.expander("How this test works"):
    st.markdown("""
    This app tests if CrewAI can successfully connect to OpenAI's API by:

    1. Setting up a CrewAI LLM with your API key
    2. Creating a simple agent with that LLM
    3. Creating a basic task for the agent (with expected_output field)
    4. Running the crew with sequential processing
    5. Measuring how long it takes to get a response

    If successful, you'll see the agent's response and timing information.
    If it fails, you'll see detailed error information to help diagnose the issue.

    Note: You no longer need langchain with the latest version of CrewAI, as they've implemented their own LLM interface.
    """)

st.markdown("---")
st.caption("Created for testing CrewAI connectivity issues")