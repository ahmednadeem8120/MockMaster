"""
evaluator_formula_test_quantitative.py — MockMaster Formula Benchmark
======================================================================
20 genuine test cases derived from Yatin Arora's CV and the Material
Planner job description.  Each test case has three calibrated answer
tiers (strong / weak / irrelevant).

The script:
  1. Loads SBERT (all-MiniLM-L6-v2) and Llama 3 via Ollama.
  2. Scores all 60 responses (20 TC × 3 tiers) with both models.
  3. Computes composite scores for 11 formula weightings:
       Formula 1  — SBERT 0.0 / LLM 1.0
       Formula 2  — SBERT 0.1 / LLM 0.9
       ...
       Formula 6  — SBERT 0.5 / LLM 0.5
       ...
       Formula 11 — SBERT 1.0 / LLM 0.0
  4. Prints 11 result tables (one per formula) to stdout.
  5. Saves all results to data/formula_test_results.json.

Run:
    python evaluator_formula_test_quantitative.py

Requirements:
    pip install sentence-transformers langchain-ollama
    Ollama must be running with llama3 pulled.
"""

import json
import os
import time

from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM

# ---------------------------------------------------------------------------
# COMPOSITE FORMULA WEIGHTINGS  (SBERT weight, LLM weight)
# ---------------------------------------------------------------------------
FORMULAS = [
    (0.0, 1.0),   # Formula  1 — pure LLM
    (0.1, 0.9),   # Formula  2
    (0.2, 0.8),   # Formula  3
    (0.3, 0.7),   # Formula  4
    (0.4, 0.6),   # Formula  5 — project default (60 LLM / 40 SBERT)
    (0.5, 0.5),   # Formula  6
    (0.6, 0.4),   # Formula  7
    (0.7, 0.3),   # Formula  8
    (0.8, 0.2),   # Formula  9
    (0.9, 0.1),   # Formula 10
    (1.0, 0.0),   # Formula 11 — pure SBERT
]

# ---------------------------------------------------------------------------
# TEST CASES  (20 questions, CV + JD grounded)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "TC-01",
        "topic": "Purchase Order Management",
        "question": (
            "Describe your process for placing and managing purchase orders, including how "
            "you prioritise replenishment when multiple materials are at risk simultaneously."
        ),
        "ideal_answer": (
            "I manage POs by first reviewing demand forecasts and current inventory levels "
            "against target stock. When multiple materials are at risk I triage by production "
            "impact, lead time and supplier reliability, prioritising materials with longest "
            "lead times or lowest safety stock first. I reschedule open POs where possible, "
            "communicate proactively with suppliers, and update stakeholders on risk status "
            "daily until supply is secured."
        ),
        "responses": {
            # STRONG — bullet-style, all key facts present; LLM may under-score -> SBERT corrects
            "strong": (
                "I triage by lead time and production impact — longest lead time materials first. "
                "I pull the open PO report daily, talk to suppliers about pull-in options, and "
                "flag any unresolvable gaps to the production scheduler by 9am. Safety stock "
                "levels drive the urgency call."
            ),
            # WEAK — vague, no specifics
            "weak": (
                "I place orders based on what we need and follow up with suppliers if things are late."
            ),
            # IRRELEVANT — wrong domain entirely
            "irrelevant": (
                "I usually handle social media scheduling on Hootsuite and track engagement "
                "metrics in Google Analytics dashboards."
            ),
        },
    },
    {
        "id": "TC-02",
        "topic": "Supply Disruption Response",
        "question": (
            "Walk me through how you have managed a supply disruption caused by an unforeseen "
            "event. What contingency steps did you take?"
        ),
        "ideal_answer": (
            "When a disruption occurs I first quantify the gap — how many days of supply are "
            "at risk and which production lines are affected. I then contact alternate suppliers "
            "or request expedites, assess if materials can be transferred from other sites, and "
            "coordinate with QA if shelf-life extensions are needed. I escalate to procurement "
            "and operations leadership with a time-bounded action plan and update daily until resolved."
        ),
        "responses": {
            "strong": (
                "During a port delay I mapped the supply gap by SKU and production day, identified "
                "two alternate domestic suppliers and got emergency quotes within 24 hours, coordinated "
                "a site transfer for the highest-priority material, and issued a daily risk update to "
                "operations until the shipment cleared."
            ),
            "weak": (
                "I call the supplier and ask them what happened and when the material will arrive."
            ),
            "irrelevant": (
                "Supply disruptions in music production usually mean re-recording sessions and "
                "rescheduling studio time with the audio engineers."
            ),
        },
    },
    {
        "id": "TC-03",
        "topic": "Inventory Optimisation",
        "question": (
            "How do you balance preventing material shortages against avoiding excess inventory, "
            "and what tools or metrics do you rely on?"
        ),
        "ideal_answer": (
            "I use target stock levels calculated from average daily usage plus safety stock based "
            "on lead time variability. I track inventory turnover, days-on-hand, and pallet counts "
            "weekly using ERP reports to flag items above or below target. For excess I work on PO "
            "rescheduling or site transfers before shelf-life becomes a risk, and for shortages I "
            "expedite or trigger safety stock replenishment rules."
        ),
        "responses": {
            "strong": (
                "I monitor days-on-hand and turnover ratio weekly. Items above max target get PO "
                "pull-outs or site transfer requests. Items below safety stock get expedited. ERP "
                "reorder point triggers handle routine replenishment so I focus manual effort on exceptions."
            ),
            "weak": (
                "I try to keep enough stock so we don't run out but not too much so it doesn't expire."
            ),
            "irrelevant": (
                "Balancing a photography portfolio means keeping a diverse range of subjects while "
                "not overwhelming clients with too many similar shots."
            ),
        },
    },
    {
        "id": "TC-04",
        "topic": "ERP and Data Analytics",
        "question": (
            "Describe your experience using ERP systems and Excel for supply chain analytics. "
            "What specific analyses do you run regularly?"
        ),
        "ideal_answer": (
            "I use ERP to generate open PO reports, inventory position reports, and purchase price "
            "variance reports. In Excel I build pivot analyses to track PPV by supplier and material, "
            "create demand vs supply dashboards with conditional formatting for at-risk items, and run "
            "forecast accuracy reconciliation on a bi-weekly basis. I also use vlookup and index-match "
            "for joining ERP exports with supplier lead time tables."
        ),
        "responses": {
            "strong": (
                "In ERP I run open PO, inventory position, and PPV reports daily. In Excel I pivot "
                "by supplier to track PPV trends, use conditional formatting to flag items outside "
                "min-max bands, and do bi-weekly forecast vs actual reconciliation using index-match "
                "to join data sets."
            ),
            "weak": (
                "I know Excel and use ERP systems at work to track orders and inventory."
            ),
            "irrelevant": (
                "I use Lightroom for photo editing and sometimes Excel to track which clients have "
                "paid their invoices."
            ),
        },
    },
    {
        "id": "TC-05",
        "topic": "Supplier Communication",
        "question": (
            "How do you manage daily communication with suppliers to ensure deliveries align "
            "with production schedules?"
        ),
        "ideal_answer": (
            "I maintain a supplier contact cadence based on lead time risk — daily for critical "
            "or at-risk items, weekly for stable suppliers. I share rolling 4-week production "
            "schedules with key suppliers, confirm delivery dates against open POs each morning, "
            "and log all commitments in a PO tracker. Any deviation from confirmed dates triggers "
            "an immediate escalation call and updated risk communication to operations."
        ),
        "responses": {
            # STRONG — fragmented delivery, all content correct; LLM may under-score -> SBERT corrects
            "strong": (
                "I send the top-10 at-risk suppliers a daily confirmation request and update the PO "
                "tracker with committed delivery dates by noon. Stable suppliers get a weekly check-in. "
                "Every deviation from a confirmed date gets escalated same day with a revised ETA "
                "sent to the production scheduler."
            ),
            "weak": (
                "I email suppliers when I need an update and they usually get back to me."
            ),
            "irrelevant": (
                "Daily communication in fitness coaching means sending your clients workout reminders "
                "and checking in on their nutrition logs every morning."
            ),
        },
    },
    {
        "id": "TC-06",
        "topic": "Forecasting and Demand Planning",
        "question": (
            "Describe how you maintain forecast accuracy and reconcile variances. What corrective "
            "actions do you take when forecast accuracy degrades?"
        ),
        "ideal_answer": (
            "I reconcile forecast vs actual consumption bi-weekly at the material level, calculate "
            "MAPE by supplier and product family, and investigate root causes of variances exceeding "
            "15%. Common causes include promotional lifts, new product launches, or production plan "
            "changes. Corrective actions include adjusting safety stock, updating replenishment "
            "parameters, or flagging the variance to the commercial team for input on the forecast model."
        ),
        "responses": {
            "strong": (
                "Bi-weekly reconciliation at material level. I calculate MAPE and flag anything over "
                "15% variance for root cause analysis — usually a promo not reflected in the baseline "
                "or a production plan change. I then adjust safety stock or replenishment parameters "
                "and document the correction."
            ),
            "weak": (
                "I check the forecast regularly and update it when the numbers look wrong."
            ),
            "irrelevant": (
                "Forecasting audience attendance for live events involves reviewing past ticket sales "
                "trends and adjusting marketing spend accordingly."
            ),
        },
    },
    {
        "id": "TC-07",
        "topic": "Cross-functional Collaboration",
        "question": (
            "Give an example of working cross-functionally with sales, operations, and procurement "
            "on a material commercialisation or cost-reduction project."
        ),
        "ideal_answer": (
            "I led a packaging material consolidation project where I worked with procurement on "
            "contract negotiations, with R&D on specification approvals, with QA on testing sign-off, "
            "and with sales on phasing the transition without disrupting customer orders. I coordinated "
            "weekly cross-functional status calls, tracked milestones in a project plan, and reported "
            "savings against target monthly. The project reduced PPV by 8% on that material family."
        ),
        "responses": {
            "strong": (
                "I ran a packaging consolidation across three suppliers — weekly cross-functional calls "
                "with procurement, R&D, QA, and sales, milestone tracking in a shared project plan, and "
                "reported monthly savings vs target. We hit an 8% PPV reduction on that material family "
                "within six months."
            ),
            "weak": (
                "I work with different departments when there is a project and we try to coordinate together."
            ),
            "irrelevant": (
                "Cross-functional collaboration in film production means aligning the director, "
                "cinematographer and costume designer on the visual aesthetic before shooting begins."
            ),
        },
    },
    {
        "id": "TC-08",
        "topic": "Purchase Price Variance",
        "question": (
            "What is purchase price variance and how do you investigate and resolve PPV discrepancies?"
        ),
        "ideal_answer": (
            "PPV is the difference between the standard cost of a material and the actual purchase "
            "price paid. I investigate PPV by pulling the ERP price variance report, identifying "
            "materials with the largest absolute and percentage variance, then checking against the "
            "contract price to determine if it is a pricing error, currency movement, or a legitimate "
            "market price change. Resolution involves either correcting the PO price if it was a data "
            "entry error, renegotiating with the supplier, or escalating to procurement for contract review."
        ),
        "responses": {
            "strong": (
                "PPV is actual price minus standard cost. I pull the ERP PPV report weekly, rank by "
                "absolute variance, then trace each outlier back to the contract rate. If it is a keying "
                "error I correct the PO. If the supplier has invoiced above contract I raise a dispute. "
                "Genuine market shifts go to procurement for contract renegotiation."
            ),
            "weak": (
                "PPV is when the price you pay is different from what you expected. I check with the "
                "supplier to fix it."
            ),
            "irrelevant": (
                "Price variance in real estate refers to the gap between asking price and final sale "
                "price, which is tracked by brokers in their CRM systems."
            ),
        },
    },
    {
        "id": "TC-09",
        "topic": "Warehouse Space Optimisation",
        "question": (
            "How have you optimised warehouse space and managed storage capacity constraints "
            "while maintaining adequate inventory levels?"
        ),
        "ideal_answer": (
            "I review pallet count reports weekly against warehouse capacity thresholds. When capacity "
            "is tight I accelerate consumption of excess materials by coordinating production scheduling "
            "changes, request early deliveries on fast-movers to clear slow-moving stock, and work with "
            "suppliers to shift to milk-run or JIT delivery on bulky materials. I also identify obsolete "
            "or near-expired stock for write-off or return to reduce footprint."
        ),
        "responses": {
            "strong": (
                "Weekly pallet count vs capacity report. When I am near the threshold I pull-in "
                "deliveries on fast movers, push out deliveries on slow movers, and flag any "
                "near-expired stock for write-off or supplier return. For bulky items I have negotiated "
                "JIT delivery windows with suppliers to reduce average dwell time."
            ),
            "weak": (
                "I try to make sure there is enough space in the warehouse by not ordering too much at once."
            ),
            "irrelevant": (
                "Warehouse space in interior design refers to the concept of negative space and how "
                "open areas between furniture pieces affect the perception of a room."
            ),
        },
    },
    {
        "id": "TC-10",
        "topic": "KPI Management",
        "question": (
            "Which supply chain KPIs do you track and how do you drive performance against targets "
            "for inventory turnover, fill rate, and expired materials?"
        ),
        "ideal_answer": (
            "I track inventory turnover, days-on-hand, purchase price variance, pallet counts, target "
            "stock levels, expired material value, and outstanding invoices. I review these weekly in "
            "a KPI dashboard and present monthly to the leadership team. When turnover is below target "
            "I analyse by material to identify slow-movers and implement pull-forward or return "
            "strategies. Expired material KPI drives proactive shelf-life reviews 90 days out from expiry."
        ),
        "responses": {
            "strong": (
                "Weekly KPI dashboard covering turnover, days-on-hand, PPV, pallet counts, and expired "
                "material value. Monthly leadership review. Slow turnover triggers material-level analysis "
                "and I push slow-movers into production planning or initiate supplier returns. Shelf-life "
                "risk gets flagged at 90-day horizon."
            ),
            "weak": (
                "I track the main KPIs and report them to my manager. If something is off I look into it."
            ),
            "irrelevant": (
                "KPIs in digital marketing include click-through rate, cost per acquisition, and return "
                "on ad spend, typically tracked in Google Analytics or HubSpot dashboards."
            ),
        },
    },
    {
        "id": "TC-11",
        "topic": "Contract Price Discrepancies",
        "question": (
            "How do you identify and resolve discrepancies between contract prices and actual invoice prices?"
        ),
        "ideal_answer": (
            "I compare the ERP standard price against the supplier invoice price for each line item on "
            "receipt. Discrepancies trigger an automated PPV report flag. I trace each discrepancy to the "
            "signed contract, check for any approved price change memos, and either reject the invoice "
            "for correction, issue a purchase order amendment if a legitimate change was missed in ERP, "
            "or escalate to the category manager if the price deviation is outside contract tolerance."
        ),
        "responses": {
            "strong": (
                "I match invoice price to ERP standard on receipt. Any gap goes on the PPV report. "
                "I trace to the contract, check for approved amendment memos, then either reject the "
                "invoice, amend the PO in ERP, or escalate to the category manager if it is a pricing "
                "dispute outside my authority."
            ),
            "weak": (
                "I check the invoice and compare it to what we agreed with the supplier and contact "
                "them if it is wrong."
            ),
            "irrelevant": (
                "Contract price discrepancies in home renovation occur when contractor quotes differ "
                "from the final bill due to scope changes or material cost increases."
            ),
        },
    },
    {
        "id": "TC-12",
        "topic": "New Material Commercialisation",
        "question": (
            "Walk me through how you manage the supply chain setup for a new material being "
            "introduced to production."
        ),
        "ideal_answer": (
            "For new material commercialisation I start by working with procurement on supplier "
            "qualification and contract setup, confirm lead times and MOQ with the supplier, set up "
            "the material master in ERP with correct replenishment parameters, and coordinate with QA "
            "on incoming inspection requirements. I then run a pilot order and validate pricing accuracy "
            "before full rollout, communicating timelines to operations and ensuring safety stock is "
            "built before the first production run."
        ),
        "responses": {
            "strong": (
                "New material setup: I work with procurement on contract and supplier qualification, "
                "then set up the material master in ERP with correct lead time, MOQ, and reorder point. "
                "QA specs for incoming inspection are confirmed before the pilot order. I validate "
                "invoice pricing after the first receipt and confirm safety stock is in place before "
                "the first scheduled production run."
            ),
            "weak": (
                "I set up the new material in the system and make sure the supplier knows we need it."
            ),
            "irrelevant": (
                "New material commercialisation in fashion design involves sourcing sustainable fabrics, "
                "presenting them to buyers, and managing minimum order quantities with textile mills."
            ),
        },
    },
    {
        "id": "TC-13",
        "topic": "Shelf-life and Expiry Risk",
        "question": (
            "How do you proactively manage shelf-life risk for perishable materials and what steps "
            "do you take when material is approaching expiry?"
        ),
        "ideal_answer": (
            "I run a shelf-life report monthly, flagging any materials with less than 90 days remaining "
            "relative to their reorder point. For at-risk materials I first check if production can "
            "consume them ahead of schedule, then contact QA about shelf-life extension testing, explore "
            "transfer to another site with faster consumption, or initiate a supplier return. If write-off "
            "is unavoidable I document the root cause and adjust safety stock or replenishment frequency "
            "to prevent recurrence."
        ),
        "responses": {
            # STRONG — fragmented but all logic is there; LLM may under-score -> SBERT corrects
            "strong": (
                "Monthly shelf-life report flags anything under 90 days. I first check production can "
                "pull the material forward. If not, QA gets a shelf-life extension request. If that "
                "fails I check if another site can absorb it or the supplier will take it back. "
                "Write-off is the last resort and triggers a root cause note and parameter adjustment."
            ),
            "weak": (
                "I check expiry dates and try to use the material before it expires. If it is close "
                "I tell my manager."
            ),
            "irrelevant": (
                "Shelf-life management for cosmetics brands involves labelling regulations, batch "
                "tracking, and consumer safety compliance managed by the regulatory affairs team."
            ),
        },
    },
    {
        "id": "TC-14",
        "topic": "Supplier Performance Management",
        "question": (
            "How do you assess and manage supplier performance, particularly for delivery "
            "reliability and quality conformance?"
        ),
        "ideal_answer": (
            "I track supplier on-time delivery rate and quality rejection rate monthly and review "
            "these in quarterly business reviews with key suppliers. When a supplier falls below the "
            "95% OTD target I issue a formal performance notice, request a corrective action plan with "
            "deadlines, and increase order confirmation frequency to daily until performance recovers. "
            "Repeat failures trigger procurement involvement for contract review or dual-sourcing evaluation."
        ),
        "responses": {
            "strong": (
                "Monthly OTD and rejection rate scorecard by supplier. Anyone below 95% OTD gets a "
                "formal performance notice and corrective action plan request. I increase confirmation "
                "cadence to daily and loop in procurement if there is no improvement within 30 days. "
                "Persistent failures go to dual-sourcing evaluation."
            ),
            "weak": (
                "I monitor how well suppliers are doing and let procurement know if there are problems."
            ),
            "irrelevant": (
                "Supplier performance in the music industry means evaluating sound engineers and studio "
                "equipment rental companies on responsiveness and audio quality."
            ),
        },
    },
    {
        "id": "TC-15",
        "topic": "Financial Purchasing Responsibilities",
        "question": (
            "Describe your experience supporting financial purchasing responsibilities such as "
            "year-end closures, invoice discrepancies, and collaboration with accounts payable."
        ),
        "ideal_answer": (
            "At year-end I reconcile all open POs to ensure they are correctly accrued or closed, "
            "work with accounts payable to clear any outstanding invoices before the financial close, "
            "and confirm pricing records are updated in ERP. Throughout the year I investigate and "
            "resolve PPV with accounts payable within five business days of flagging, maintain GR-IR "
            "reconciliation, and ensure goods receipts are posted same day as physical delivery to "
            "avoid accrual timing issues."
        ),
        "responses": {
            "strong": (
                "Year-end I reconcile all open POs and clear outstanding invoices with AP before the "
                "close date. I confirm ERP pricing is updated and accrue for any goods received but "
                "not yet invoiced. During the year I target 5-day resolution on PPV disputes and post "
                "goods receipts same day as delivery for clean GR-IR reconciliation."
            ),
            "weak": (
                "I help with invoices and make sure things are paid on time. At year end I help close out orders."
            ),
            "irrelevant": (
                "Year-end financial close in a law firm involves billing unbilled hours, reconciling "
                "client retainers, and filing regulatory returns with the bar association."
            ),
        },
    },
    {
        "id": "TC-16",
        "topic": "Replenishment Strategy",
        "question": (
            "Explain the difference between min-max replenishment and reorder point planning, "
            "and when you would choose each approach."
        ),
        "ideal_answer": (
            "Min-max replenishment triggers an order when stock falls to the minimum level and orders "
            "up to the maximum, making it simple and suitable for low-variability items with stable "
            "demand. Reorder point planning calculates a dynamic trigger based on average daily usage "
            "multiplied by lead time plus safety stock, making it more responsive to demand variability. "
            "I use min-max for commodity materials with consistent consumption and reorder point planning "
            "for materials with variable demand or longer lead times where safety stock calculation is critical."
        ),
        "responses": {
            "strong": (
                "Min-max is a fixed band — order when you hit the min, order up to the max. Good for "
                "stable low-variability materials. Reorder point is dynamic — average daily usage times "
                "lead time plus safety stock. Better for variable demand or long lead time materials. "
                "I use min-max for commodity packaging and ROP for seasonal or variable-consumption ingredients."
            ),
            "weak": (
                "Min-max means you keep stock between a minimum and maximum amount. Reorder point "
                "means you order when stock gets low."
            ),
            "irrelevant": (
                "In graphic design, min-max refers to the minimum and maximum font sizes used in "
                "responsive typography systems to maintain readability across screen sizes."
            ),
        },
    },
    {
        "id": "TC-17",
        "topic": "Cost Reduction Initiatives",
        "question": (
            "Describe a cost reduction initiative you led or contributed to in a supply chain "
            "or purchasing context."
        ),
        "ideal_answer": (
            "I led a PO rescheduling initiative that reduced excess inventory carrying costs by "
            "consolidating deliveries for slow-moving materials and extending delivery intervals "
            "where shelf-life allowed. I identified the opportunity through a turnover analysis, "
            "built a business case with the projected carrying cost savings, got buy-in from "
            "operations and finance, and implemented new delivery frequency agreements with five "
            "suppliers. The result was a 12% reduction in average days-on-hand for that material group."
        ),
        "responses": {
            "strong": (
                "I ran a delivery frequency consolidation on slow-moving packaging — identified it "
                "through turnover analysis, built a carrying cost savings model, got operations and "
                "finance sign-off, then renegotiated delivery intervals with five suppliers. We cut "
                "average days-on-hand by 12% on that material group within a quarter."
            ),
            "weak": (
                "I helped save money by ordering less and negotiating better prices with some suppliers."
            ),
            "irrelevant": (
                "Cost reduction in film production typically involves cutting locations, reducing "
                "shooting days, or replacing name actors with emerging talent to stay within budget."
            ),
        },
    },
    {
        "id": "TC-18",
        "topic": "SAP and Systems Knowledge",
        "question": (
            "What is your experience with SAP or similar ERP systems in a purchasing or materials "
            "planning role? What transactions or reports do you rely on most?"
        ),
        "ideal_answer": (
            "I have used ERP systems extensively for PO creation and management, inventory reporting, "
            "and PPV analysis. Key transactions I use regularly include the open PO report, goods "
            "receipt confirmation, stock overview, and the purchase price variance report. I also use "
            "the material requirements planning run output to review system-generated replenishment "
            "proposals and adjust where the system does not account for supplier constraints or "
            "shelf-life risk."
        ),
        "responses": {
            "strong": (
                "I use ERP daily for PO management and inventory reporting — open PO report, stock "
                "overview, GR confirmation, and PPV report are my daily transactions. I also review "
                "the MRP proposal output each morning to catch any system-generated orders that need "
                "manual adjustment for supplier lead time exceptions or shelf-life constraints."
            ),
            "weak": (
                "I use SAP at work for ordering and checking stock levels. I know how to create POs "
                "and check inventory."
            ),
            "irrelevant": (
                "SAP is also used in HR departments for payroll processing, leave management, and "
                "organisational chart maintenance across large multinational companies."
            ),
        },
    },
    {
        "id": "TC-19",
        "topic": "Stakeholder Communication",
        "question": (
            "How do you communicate material shortage risks to internal stakeholders and what "
            "information do you include?"
        ),
        "ideal_answer": (
            "I issue a daily risk report to operations and production scheduling covering any material "
            "with less than five days of supply at current consumption rate, the expected stockout date, "
            "the root cause, the mitigation actions in progress, and the expected resolution date. For "
            "high-severity shortages I also escalate to the supply chain manager and plant director "
            "with a verbal briefing and a written summary covering business impact and worst-case scenario."
        ),
        "responses": {
            "strong": (
                "Daily risk report to operations and production covering material, days of supply "
                "remaining, expected stockout date, root cause, mitigation steps, and ETA to resolution. "
                "Anything under three days gets a verbal escalation to the supply chain manager and "
                "plant director with a worst-case impact statement."
            ),
            "weak": (
                "I tell operations when we might run out of something and what I am doing to fix it."
            ),
            "irrelevant": (
                "Communicating risks to stakeholders in a software project means writing a risk "
                "register in Jira and presenting it during the weekly engineering stand-up meeting."
            ),
        },
    },
    {
        "id": "TC-20",
        "topic": "Continuous Improvement",
        "question": (
            "Describe an improvement you identified and implemented in a purchasing or supply chain "
            "process, and what the measurable outcome was."
        ),
        "ideal_answer": (
            "I identified that our PO confirmation process was entirely manual and took three hours "
            "daily across the team. I designed an Excel macro that pulled open POs from ERP, flagged "
            "any where the confirmed delivery date was within two days of the required date, and "
            "auto-populated the supplier email template. This cut the daily confirmation process from "
            "three hours to 45 minutes and reduced late delivery surprises by 30% over the following quarter."
        ),
        "responses": {
            "strong": (
                "Our PO confirmation process was three hours of manual work daily. I built an Excel "
                "macro that extracted open POs from ERP, flagged at-risk delivery dates, and "
                "pre-populated supplier emails. The process dropped to 45 minutes and late delivery "
                "surprises fell 30% in the next quarter."
            ),
            "weak": (
                "I found a better way to track orders using a spreadsheet that I shared with the team."
            ),
            "irrelevant": (
                "Process improvement in a restaurant kitchen means standardising recipes, reducing "
                "plating time, and using prep lists to minimise food waste during service."
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------------------------

def compute_sbert_score(model, candidate: str, ideal: str) -> float:
    """Cosine similarity scaled to 0-10."""
    if not candidate.strip() or not ideal.strip():
        return 0.0
    embeddings = model.encode([candidate, ideal], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(max(0.0, similarity.item())) * 10, 2)


def compute_llm_score(llm, question: str, ideal: str, candidate: str, word_count: int) -> int:
    """Ask Llama 3 to score the answer 1-10."""
    is_too_short = word_count < 15
    prompt = f"""
You are a strict supply chain interviewer evaluating a spoken answer.
Judge content purely on its own merit — no semantic scores are provided.

Question: "{question}"
Ideal answer outline: "{ideal}"
Candidate answered: "{candidate}"
Word count: {word_count}

SCORING RULES:
- 1-2: Fewer than 15 words or completely off-topic.
- 3-4: Correct topic but vague — no specific methods, tools, or steps named.
- 5-6: Covers the key idea but lacks depth or misses important details.
- 7-8: Strong — names concrete tools, methods, or approaches with reasoning.
- 9-10: Exceptional — technically deep with specific examples or metrics.
Be strict. Most answers should score 4-6.

Respond ONLY with a JSON object: {{"score": <integer 1-10>}}
"""
    try:
        raw = llm.invoke(prompt)
        data = json.loads(raw)
        score = int(data.get("score", 5))
        score = max(1, min(10, score))
        if is_too_short:
            score = min(score, 2)
        return score
    except Exception:
        return 2 if is_too_short else 5


def composite(sbert_score: float, llm_score: int, sbert_w: float, llm_w: float) -> float:
    """Weighted composite of SBERT (0-10) and LLM (0-10)."""
    return round(sbert_score * sbert_w + llm_score * llm_w, 2)


# ---------------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------------

def print_formula_table(formula_idx: int, sw: float, lw: float, results: list):
    """Print one result table for a given formula weighting."""
    label = f"Formula {formula_idx+1:>2}  —  SBERT {sw:.1f} / LLM {lw:.1f}"
    if abs(sw - 0.4) < 0.001:
        label += "  [PROJECT DEFAULT]"

    col_w = [6, 34, 11, 12, 12, 12]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr_fmt = "|{:^6}|{:<34}|{:^11}|{:^12}|{:^12}|{:^12}|"
    row_fmt  = "|{:^6}|{:<34}|{:^11}|{:^12.2f}|{:^12}|{:^12.2f}|"

    print("\n" + "=" * 92)
    print(f"  {label}")
    print("=" * 92)
    print(sep)
    print(hdr_fmt.format("ID", "Topic", "Tier", "SBERT /10", "LLM /10", "Composite /10"))
    print(sep)

    levels = ["strong", "weak", "irrelevant"]
    for r in results:
        comp = composite(r["sbert_score"], r["llm_score"], sw, lw)
        print(row_fmt.format(
            r["id"],
            r["topic"][:33],
            r["level"],
            r["sbert_score"],
            r["llm_score"],
            comp,
        ))
    print(sep)

    # Summary stats
    print()
    for level in levels:
        sub = [r for r in results if r["level"] == level]
        avg_s = sum(r["sbert_score"] for r in sub) / len(sub)
        avg_l = sum(r["llm_score"]   for r in sub) / len(sub)
        avg_c = sum(composite(r["sbert_score"], r["llm_score"], sw, lw) for r in sub) / len(sub)
        print(f"  Average ({level:<10}): SBERT={avg_s:.2f}  LLM={avg_l:.2f}  Composite={avg_c:.2f}")

    strong_avgs = [composite(r["sbert_score"], r["llm_score"], sw, lw)
                   for r in results if r["level"] == "strong"]
    irrel_avgs  = [composite(r["sbert_score"], r["llm_score"], sw, lw)
                   for r in results if r["level"] == "irrelevant"]
    weak_avgs   = [composite(r["sbert_score"], r["llm_score"], sw, lw)
                   for r in results if r["level"] == "weak"]
    gap_si = round(sum(strong_avgs)/len(strong_avgs) - sum(irrel_avgs)/len(irrel_avgs), 2)
    gap_sw = round(sum(strong_avgs)/len(strong_avgs) - sum(weak_avgs)/len(weak_avgs),   2)

    # Wrong verdict analysis for this formula
    wrong = corrected = 0
    for r in results:
        lv, llm, sbert = r["level"], r["llm_score"], r["sbert_score"]
        comp_val = composite(sbert, llm, sw, lw)
        if lv == "strong" and llm < 6:
            wrong += 1
            if sbert >= 6 and comp_val > llm:
                corrected += 1
        elif lv == "weak" and llm >= 6:
            wrong += 1
            if sbert < 5 and comp_val < llm:
                corrected += 1

    rate = round(corrected / max(1, wrong) * 100)
    print(f"\n  Discrimination gap (Strong vs Irrelevant): {gap_si}/10")
    print(f"  Discrimination gap (Strong vs Weak)      : {gap_sw}/10")
    print(f"  LLM wrong verdicts  : {wrong} / {len(results)}")
    print(f"  Corrected by SBERT  : {corrected} ({rate}%)")


def save_results(all_formula_results: list):
    """Save full results for all formulas to JSON."""
    os.makedirs("data", exist_ok=True)
    path = "data/formula_test_results.json"
    with open(path, "w") as f:
        json.dump(all_formula_results, f, indent=2)
    print(f"\n  All results saved to {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run():
    print("=" * 92)
    print("  MockMaster — EVALUATOR FORMULA TEST (QUANTITATIVE)")
    print("  20 Test Cases × 3 Tiers × 11 Composite Formula Weightings")
    print("=" * 92)

    print("\nLoading SBERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SBERT ready.")

    print("Connecting to Llama 3 via Ollama...")
    llm_model = OllamaLLM(model="llama3", temperature=0.1, format="json")
    print("LLM ready.\n")

    # Step 1: Score every response once (scores are formula-independent)
    raw_results = []
    total = len(TEST_CASES) * 3
    done  = 0

    for tc in TEST_CASES:
        for level, answer in tc["responses"].items():
            done += 1
            print(f"  [{done:>2}/{total}] {tc['id']}  {level:<10}", end=" ", flush=True)
            t0 = time.time()

            sbert_score = compute_sbert_score(sbert_model, answer, tc["ideal_answer"])
            word_count  = len(answer.split())
            llm_score   = compute_llm_score(
                llm_model, tc["question"], tc["ideal_answer"], answer, word_count
            )

            elapsed = round(time.time() - t0, 1)
            print(f"SBERT={sbert_score:<5}  LLM={llm_score}  ({elapsed}s)")

            raw_results.append({
                "id":           tc["id"],
                "topic":        tc["topic"],
                "question":     tc["question"],
                "level":        level,
                "ideal_answer": tc["ideal_answer"],
                "answer":       answer,
                "sbert_score":  sbert_score,
                "llm_score":    llm_score,
            })

    # Step 2: Print one table per formula and accumulate JSON output
    all_formula_results = []

    for idx, (sw, lw) in enumerate(FORMULAS):
        print_formula_table(idx, sw, lw, raw_results)

        formula_entry = {
            "formula_index":   idx + 1,
            "sbert_weight":    sw,
            "llm_weight":      lw,
            "is_project_default": abs(sw - 0.4) < 0.001,
            "rows": [],
        }
        for r in raw_results:
            comp = composite(r["sbert_score"], r["llm_score"], sw, lw)
            formula_entry["rows"].append({**r, "composite": comp})

        all_formula_results.append(formula_entry)

    save_results(all_formula_results)

    print("\n" + "=" * 92)
    print("  Run complete.  11 tables printed.  Results saved to data/formula_test_results.json")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    run()
