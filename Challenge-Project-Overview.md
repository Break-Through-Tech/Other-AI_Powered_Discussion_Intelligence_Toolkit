---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. In order for your project to be finalized and assigned to a team, please:
>
> 1. **Send us your GitHub username** so we can add you as a Collaborator to this repo, which will enable you to make edits. If you don't have a username, you can create a free account [here](https://github.com/signup). Once you are ready to share your username, simply reply to the email that sent you to this repo. Once we receive your GitHub username, you will get an email inviting you to join this repo as a Collaborator and can begin making edits. 
> 2. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").  
> 3. **Add your dataset** to the [data folder](data) in this repo.
> 4. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text. 
> 
> ---
>
### 🔍 [SME Feedback from the BTT Evaluation Team](https://github.com/Break-Through-Tech/Other_AI_Powered_Discussion_Intelligence_Toolkit/blob/main/SME_Feedback.md)

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The tech stack is entirely centered on Python, leveraging popular libraries for machine learning and NLP. |
| Data Readiness | 🟢  | The datasets are indicated to be under 1GB, suggesting they are manageable and ready for use without extensive cleaning. |
| Resource Check | 🟢  | The project utilizes Google Colab, which is accessible to students. Although it includes GPU access, this is available through the free tier. |

**Student Fit Score:** 8/10  
**Technical Depth Score:** 7/10  
**Overall Recommendation:** APPROVE

**Advisor Feedback Draft:**
This project presents a strong application of modern NLP techniques. However, consider simplifying some aspects of the model training process to ensure students can fully engage with the content. Additionally, clarify how clustering results will be interpreted and used in the context of structured reports. Encourage focusing on robust evaluation metrics for the models tracked over time.

🔍 Detail SME Project Feedback
---

# Discussion Intelligence Toolkit

**Company / Org:** Break Through Tech  
**Challenge Advisor:** Tom Mathews, tom.mathews@nyu.edu  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Break Through Tech

Break Through Tech is dedicated to increasing the representation of women and underrepresented groups in technology through training, job placement, and community building activities. Our initiatives span across various industries and aim to equip individuals with the necessary skills to thrive in a tech-focused environment.

---

## 🎯 The Challenge

### Project Summary
An open-source Python toolkit designed to transform raw asynchronous conversations (such as Discord threads or GitHub conversations) into structured insight reports. The toolkit uses transformer-based classifiers for discourse act tagging, clustering for topic discovery, and retrieval-augmented LLM synthesis to produce reports that highlight key arguments and consensus.

### Success Criteria
The production of structured insight reports that highlight high-quality arguments and link them to supporting comments, along with model evaluation results and a demo-ready portfolio artifact.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month      | Milestone                  | Key Activities                                                      |
|------------|----------------------------|--------------------------------------------------------------------|
| **September** | Data Understanding         | Explore dataset, handle missing values, document findings         |
| **October**   | Model Development          | Train baseline model, experiment with approaches, iterate         |
| **November**  | Evaluation & Presentation   | Finalize model, prepare presentation, document results            |
| **December**  | Implementation & Wrap Up    | Produce final insights, deliver demo-ready artifact              |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** [e.g., Dataset name and where it's from]  
**Format:** CSV  
**Size:** under 1gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- The project utilizes datasets consisting of approximately 115,000 utterances for discourse classification and 43,000 utterances for argument quality scoring, potentially sourced from Discord exports, forum dumps, or podcast transcripts.
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP

**Recommended Libraries:**
- Python
- DistilBERT
- TensorFlow
- Keras
- Hugging Face (TF backend)
- scikit-learn
- NumPy
- sentence-transformers
- FAISS or Chroma
- Google Colab

**Evaluation Metrics:**
- Accuracy
- Precision/Recall
- BLEU score

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Link to an article or blog post about the problem domain]
- [Link to an industry report or case study]

**Technical Tutorials:**
- [Link to a free tutorial on the ML technique(s) involved]
- [Link to documentation for a key library or tool]

**Code Examples:**
- [Link to a relevant GitHub repo]
- [Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace) or email  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab, VS Code
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting, scheduled for the week of August 24th (Break Through Tech's Bridge to Studio - Session B).

---
