# Meridian Biopharma — third-party data extract

Synthetic data for the Duke MIDS 2026 capstone. **Meridian Biopharma plc is a fictional company.**
Countries, risk indices and regulatory regimes are real; every entity, person, address, engagement
and payment in these files is synthetic. No real third party, employee or individual is represented.

## How to read this extract

These are **operational extracts**, not a graph: vendor-master style tables plus two external
registry feeds. Building the canonical graph from them is part of the exercise.

**Time.** Two different clocks appear in the data and must be kept apart:

| Field | Meaning |
|---|---|
| `valid_from` / `valid_to`, `start_date` / `end_date` | when the relationship was true in the world (valid time) |
| `observed_at` | when the organisation recorded it (observation time) |
| `created_date` | when the vendor record was created in the current system, which is not the onboarding date for records migrated in the 2022 ERP move |

Nothing in the extract is observed after the snapshot date **2026-06-30**. A `valid_to` that is
null means "still open at the snapshot", not "never ended". When you build a graph as of time *t*,
filter on `observed_at <= t` as well as validity, or you will leak the future into the past.

**Unknown vs. none.** A null in `sponsor_employee_id`, `address_type`, `lei`, `registry_no` or an
ownership path means *not recorded*, not *does not exist*. Registry coverage varies by country:
see `country_risk_context` (index `BO_REGISTER_ACCESS`).

**Identifiers.** All IDs are random tokens. They carry no ordering, no age and no meaning; two
records with adjacent IDs are unrelated. Vendors can appear more than once in `third_parties`
(duplicate master records), and resolving them is part of the work.

**Money.** `amount_usd` is a monthly aggregate in USD converted at a fixed synthetic rate; there
are no invoice-, payment- or bank-level records anywhere in this extract.


## Tables

| Table | Rows | Grain | What it is |
|---|---|---|---|
| `third_parties` | 21,260 | record | One row per vendor-master record. |
| `tp_due_diligence` | 21,260 | record | Record-level due-diligence outcome captured at onboarding. |
| `tp_address_history` | 23,258 | record x address x validity period | Addresses as they were recorded, in raw free text. |
| `tp_services` | 37,723 | record x category x validity period | Service categories a third party is registered to provide. |
| `tp_engagements` | 56,086 | engagement | Contracts and statements of work. |
| `tp_engagement_approvals` | 91,808 | approval | Approval events on engagements: the financial approval required by the delegation of authority, plus a compliance review for HCP-facing, government-facing and intermediary categories. |
| `tp_payments_monthly` | 755,740 | record x engagement x month x expense classification x channel | Monthly aggregated spend per engagement and classification. |
| `tp_institution_links` | 9,127 | link | Links between third parties and healthcare institutions: which distributors serve which hospitals, which sites belong to which institution, and which institution an individual expert is affiliated with. |
| `healthcare_institutions` | 1,789 | institution | Hospitals, academic medical centres, clinics, health authorities and patient organisations. |
| `employees` | 4,181 | employee | Internal employees who sponsor or approve third-party engagements. |
| `employee_roles` | 4,181 | employee x role period | Role and scope per employee, with the dates the role was held. |
| `employee_delegations` | 6 | delegation | Documented temporary transfers of approval authority (leave cover, vacancies). |
| `markets` | 74 | market | Market reference data. |
| `paying_entities` | 74 | paying entity | Meridian affiliates that pay third parties, one per market. |
| `service_categories` | 43 | category | Service taxonomy with the v1 to v2 crosswalk and the risk flags that drive compliance review. |
| `approval_policy_matrix` | 5 | value band | Delegation of authority: which value band needs which approver level, and when a compliance review is also required. |
| `fmv_rate_cards` | 144 | card version x country band x expert role x activity | Fair-market-value caps for paying external experts, by role, activity and country band. |
| `country_risk_context` | 1,120 | country x index x edition | Dated country context from public indices. |
| `labelled_cases_TRAINING_SET` | 0 | planted case | PRACTICE LABELS, starter set only. |
| `business_events` | 10 | event | Company events that explain bursts, hubs and discontinuities in the data. |

### `third_parties`

One row per vendor-master record. A real third party may have more than one record (duplicates created over the years); `tax_id`, address and name variants are your evidence for merging them.

**Grain:** record  ·  **Rows:** 21,260

| Column | Description |
|---|---|
| `record_id` | Vendor-master record ID (random token). Primary key of this table. |
| `party_type` | organization | individual. Individuals are external experts paid to perform services (HCP consultants, investigators, writers, regulatory experts, patient experts). |
| `legal_name` | Latin-script name as held in the vendor master. Registry feeds may hold the same entity in its native script. |
| `tax_id` | Synthetic national tax identifier. Shared tax IDs are strong evidence of duplicate records. |
| `lei` | Legal Entity Identifier where the vendor has one (~25% of vendors). Join key to registry_L_parents.csv. |
| `registry_no` | Company-register number where recorded. Join key to registry_R statements. |
| `country` | Operating country (ISO 3166-1 alpha-2), joins markets.market_id. |
| `status` | active | terminated as at the snapshot. |
| `onboarding_date` | When the third party was first onboarded (may predate the current system). |
| `created_date` | When this record was created in the current system. Equals the ERP migration date for migrated records. |
| `terminated_date` | Termination date, null if still active. |
| `declared_parent_group` | Corporate group declared during onboarding. Populated for ~70% of group members; absence does not prove independence. |
| `phone` | Contact telephone number. |
| `email_domain` | Contact email domain. |
| `remit_to_country` | Country payments are remitted to. Differs from `country` for multinationals and shared-service structures. |
| `expert_type` | For individuals: the kind of external expert (hcp_consultant, investigator_kol, writer_translator, regulatory_access, patient_expert, other_consultant). |

### `tp_due_diligence`

Record-level due-diligence outcome captured at onboarding. These are the answers given and the checks run at the time — they are not a verdict.

**Grain:** record  ·  **Rows:** 21,260

| Column | Description |
|---|---|
| `dd_level` | basic | enhanced. Enhanced is risk-triggered (category flags, country tier). |
| `dd_completed` | Date due diligence was completed. |
| `dd_refresh_due` | Date the next refresh is due (1 year enhanced, 3 years basic). |
| `q_gov_ownership` | Questionnaire answer: is the third party state-owned or state-controlled? |
| `q_pep_connected` | Questionnaire answer: is an owner or controller a politically exposed person or a close associate? |
| `q_public_official_interaction` | Questionnaire answer: will the third party interact with public officials on the company's behalf? |
| `q_former_public_official` | Questionnaire answer (individuals): is the expert a former public official? |
| `screening_result` | clear | potential_match_cleared (a sanctions/adverse-media screening hit that was reviewed and cleared). |
| `adverse_media` | Adverse-media flag raised during screening. |
| `risk_rating` | low | medium | high risk rating assigned at onboarding. |
| `record_id` | Joins third_parties.record_id. |

### `tp_address_history`

Addresses as they were recorded, in raw free text. The same physical address can appear with different formatting, abbreviations, casing, field order and the occasional typo; normalising them is part of the exercise.

**Grain:** record x address x validity period  ·  **Rows:** 23,258

| Column | Description |
|---|---|
| `record_id` | Joins third_parties.record_id. |
| `raw_address` | Address exactly as held in the system. |
| `address_country` | ISO country of the address. |
| `address_type` | commercial | residential | coworking | corporate_hq | hospital_campus | registered_agent | virtual_office | po_box | industrial. Populated for ~60% of rows; null means not classified. |
| `address_role` | registered | correspondence. |
| `valid_from` | Start of the period the address was in use. |
| `valid_to` | End of the period; null means still in use at the snapshot. |
| `observed_at` | When the address was recorded. |

### `tp_services`

Service categories a third party is registered to provide. Many vendors provide several. Categories were re-coded in January 2024, so rows carry the taxonomy version that applied.

**Grain:** record x category x validity period  ·  **Rows:** 37,723

| Column | Description |
|---|---|
| `record_id` | Joins third_parties.record_id. |
| `category_code` | Service category code in the stated taxonomy version; joins service_categories. |
| `taxonomy_version` | v1 (before 2024-01-01) or v2. Use service_categories to map between them. |
| `valid_from` | Start of the period the category applied. |
| `valid_to` | End of the period; null means still current. |

### `tp_engagements`

Contracts and statements of work. Sponsors and approvers attach HERE, not to the vendor: a multi-service vendor can have a different sponsor per service line.

**Grain:** engagement  ·  **Rows:** 56,086

| Column | Description |
|---|---|
| `engagement_id` | Engagement ID (random token). Primary key. |
| `record_id` | The contracted third party; joins third_parties.record_id. |
| `service_category` | Service category of this engagement, in `taxonomy_version`. |
| `taxonomy_version` | v1 or v2, per the engagement start date. |
| `business_unit` | SPECIALTY | VACCINES | GENMED | RND | SUPPLY | CORPORATE | CONSUMER (pre-2022 demerger). |
| `market` | Market the engagement is booked in. Differs from the vendor's country for cross-border engagements. |
| `contract_type` | msa | sow | fee_for_service | grant_agreement | sponsorship_agreement | distribution_agreement | supply_agreement | agency_agreement | clinical_trial_agreement | none. |
| `start_date` | Engagement start. |
| `end_date` | Engagement end; null means open at the snapshot. |
| `value_band` | Contract value band 0-4, per approval_policy_matrix. Band, not amount, is what drives approval routing. |
| `sponsor_employee_id` | Internal requester; joins employees.employee_id. Null where not recorded (~3%). |
| `observed_at` | When the engagement was recorded. |

### `tp_engagement_approvals`

Approval events on engagements: the financial approval required by the delegation of authority, plus a compliance review for HCP-facing, government-facing and intermediary categories.

**Grain:** approval  ·  **Rows:** 91,808

| Column | Description |
|---|---|
| `approval_id` | Approval ID (random token). Primary key. |
| `engagement_id` | Joins tp_engagements.engagement_id. |
| `approver_employee_id` | Approver; joins employees.employee_id. Null where not recorded (~5%). |
| `approval_type` | financial | compliance_review. |
| `approval_date` | Date of the approval. |
| `observed_at` | When the approval was recorded. |

### `tp_payments_monthly`

Monthly aggregated spend per engagement and classification. No invoice- or payment-level detail. A vendor's spend can legitimately span several classifications when it provides several services.

**Grain:** record x engagement x month x expense classification x channel  ·  **Rows:** 755,740

| Column | Description |
|---|---|
| `record_id` | Joins third_parties.record_id. |
| `engagement_id` | Joins tp_engagements.engagement_id. |
| `month` | Calendar month (first day). |
| `paying_entity_id` | Meridian affiliate that paid; joins paying_entities. |
| `paying_market` | Market of the paying affiliate. |
| `vendor_country` | Operating country of the vendor, repeated here for convenience. |
| `service_category` | Service line the spend belongs to. |
| `expense_classification` | How the spend was booked in the ledger: consulting_fees, honoraria, grants_donations, sponsorships, travel_hospitality, marketing_services, distribution_fees, commissions, goods_materials, clinical_services, logistics, professional_services, government_fees. |
| `tov_category` | Transfer-of-value disclosure category where the spend is HCP/HCO-related (fees_for_service_consultancy, sponsorship_of_events, donations_and_grants, travel_accommodation, research_and_development, other_hcp_related); not_applicable otherwise. |
| `channel` | po_backed | non_po_invoice | expense_reimbursement | p_card. |
| `invoice_count` | Invoices in the month. |
| `payment_count` | Payments in the month. |
| `amount_usd` | Total paid in the month, USD (fixed synthetic FX). |
| `amount_local` | Same amount in the paying affiliate's currency. |
| `currency` | Currency of `amount_local`. |
| `max_invoice_band` | Band (0-4, per approval_policy_matrix) of the largest single invoice in the month. |
| `round_amount_share` | Share of invoices in the month with round amounts. |
| `manual_payment_count` | Payments made outside the standard purchase-to-pay flow. |

### `tp_institution_links`

Links between third parties and healthcare institutions: which distributors serve which hospitals, which sites belong to which institution, and which institution an individual expert is affiliated with.

**Grain:** link  ·  **Rows:** 9,127

| Column | Description |
|---|---|
| `link_id` | Link ID (random token). |
| `record_id` | Joins third_parties.record_id. |
| `institution_id` | Joins healthcare_institutions.institution_id. |
| `link_type` | SERVES (distributor/agent serves the institution) | SITE_OF (investigator site belongs to it) | AFFILIATED_WITH (individual expert is employed/affiliated there). |
| `valid_from` | Start of the link. |
| `valid_to` | End of the link; null means current. |

### `healthcare_institutions`

Hospitals, academic medical centres, clinics, health authorities and patient organisations. In many countries staff of public institutions are public officials, which is why `public_sector` matters.

**Grain:** institution  ·  **Rows:** 1,789

| Column | Description |
|---|---|
| `institution_id` | Institution ID (random token). |
| `name` | Institution name. |
| `country` | ISO country. |
| `inst_type` | hospital | academic_medical_center | clinic | health_authority | patient_organization. |
| `public_sector` | True if publicly owned or controlled. |

### `employees`

Internal employees who sponsor or approve third-party engagements. Pseudonymous.

**Grain:** employee  ·  **Rows:** 4,181

| Column | Description |
|---|---|
| `employee_id` | Employee ID (random token). |
| `name` | Synthetic employee name. |

### `employee_roles`

Role and scope per employee, with the dates the role was held. Scope is what makes a large portfolio explainable: a central category manager legitimately covers every market for one category.

**Grain:** employee x role period  ·  **Rows:** 4,181

| Column | Description |
|---|---|
| `employee_id` | Joins employees.employee_id. |
| `role` | brand_manager, medical_lead, market_access_lead, clinical_ops, supply_buyer, other_sponsor, market_gm, category_manager, procurement_ss, compliance_reviewer, finance_approver_l1/l2/l3, regional_finance_l3, system_default. |
| `franchise` | Franchise for commercial roles (SPECIALTY, VACCINES, GENMED), null otherwise. |
| `home_market` | Market the role sits in. |
| `scope_markets` | Markets in scope, ';' separated. '*' = global, 'R:<region>' = a region, 'S:<subregion>' = a subregion. |
| `scope_categories` | Category parents in scope, ';' separated ('*' = all). Joins service_categories.parent. |
| `approver_level` | Delegation-of-authority level the role can approve (0 = not an approver). |
| `valid_from` | Start of the role period. |
| `valid_to` | End of the role period; null means current. |

### `employee_delegations`

Documented temporary transfers of approval authority (leave cover, vacancies). Explains approval concentration in a specific window.

**Grain:** delegation  ·  **Rows:** 6

| Column | Description |
|---|---|
| `from_employee_id` | Employee whose authority was delegated. |
| `to_employee_id` | Employee who exercised it. |
| `start` | Start of the delegation. |
| `end` | End of the delegation. |
| `reason` | leave_cover | parental_leave | vacancy. |

### `markets`

Market reference data. Region and subregion define peer groups for market-level comparisons.

**Grain:** market  ·  **Rows:** 74

| Column | Description |
|---|---|
| `market_id` | ISO country code. |
| `name` | Country name. |
| `region` | US | EUROPE | JAPAN_CHINA | INTL. |
| `subregion` | Finer grouping (EU_WEST, LATAM, MEA, APAC, CIS, ...). |

### `paying_entities`

Meridian affiliates that pay third parties, one per market.

**Grain:** paying entity  ·  **Rows:** 74

| Column | Description |
|---|---|
| `paying_entity_id` | Paying entity ID. |
| `name` | Affiliate name. |
| `market_id` | Joins markets.market_id. |

### `service_categories`

Service taxonomy with the v1 to v2 crosswalk and the risk flags that drive compliance review.

**Grain:** category  ·  **Rows:** 43

| Column | Description |
|---|---|
| `category_code_v2` | Current category code. |
| `parent` | Parent category (COM, HCP, EVT, MED, RND, SCM, PAT, GNA). |
| `name` | Human-readable category name. |
| `category_code_v1` | Pre-2024 code for the same service. Several v2 codes can map to one v1 code. |
| `flags` | ';' separated risk flags: hcp (HCP-interacting), gov (government-interacting), int (acts as an intermediary), flow (high cash or physical flow). |

### `approval_policy_matrix`

Delegation of authority: which value band needs which approver level, and when a compliance review is also required.

**Grain:** value band  ·  **Rows:** 5

| Column | Description |
|---|---|
| `value_band` | Band 0-4 used in tp_engagements.value_band and tp_payments_monthly.max_invoice_band. |
| `min_usd` | Lower bound of the band (USD). |
| `max_usd` | Upper bound; null for the top band. |
| `required_approver_level` | Minimum approver level required (joins employee_roles.approver_level). |
| `compliance_review_if_flags` | Category flags that additionally require a compliance review. |

### `fmv_rate_cards`

Fair-market-value caps for paying external experts, by role, activity and country band. Cards are refreshed periodically, so check the version in force on the engagement date. Fees above the applicable cap exist in the data and are worth finding.

**Grain:** card version x country band x expert role x activity  ·  **Rows:** 144

| Column | Description |
|---|---|
| `card_version` | Rate-card version (v1 to end-2023, v2 from 2024). |
| `effective_from` | First date this version applied. |
| `effective_to` | Last date it applied; null means current. |
| `country_band` | US | EU_JP | EMERGING. Bands group markets by typical fee level. |
| `expert_role` | Expert role the cap applies to; joins third_parties.expert_type. |
| `activity_type` | advisory_board_day | steering_committee_day | speaker_engagement | consulting_hour. |
| `unit` | What the rate is per: day, engagement or hour. |
| `max_rate_usd` | Maximum rate per unit under this card. |
| `annual_cap_usd` | Maximum total fees per expert per year under this card. |

### `country_risk_context`

Dated country context from public indices. Values are attached with the edition and source they came from; they are context for interpretation, never a label.

**Grain:** country x index x edition  ·  **Rows:** 1,120

| Column | Description |
|---|---|
| `iso2` | ISO country code. |
| `index` | CPI (Transparency International Corruption Perceptions Index) | SANCTIONS (OFAC country programmes, with the legal basis in `source`) | BO_REGISTER_ACCESS (how accessible the beneficial-ownership register is), plus FSI (Financial Secrecy Index), FATF (list status) and SANCTIONS (programme level) when those sources were loaded for this extract. |
| `edition_year` | Edition/year the value belongs to. |
| `as_of_date` | Date the edition took effect for analysis purposes. Use the edition that was current at the time you are analysing. |
| `value` | Numeric score where the index has one. |
| `tier` | Banded value: CPI 1 (least risk) to 4; FSI tercile; FATF/SANCTIONS status; BO_REGISTER_ACCESS high/medium/low. |
| `source` | Source of the value. |
| `valid_to` | For listing-based indices (FATF, SANCTIONS): when the listing ended; null means still in force. Other indices use `as_of_date` and the next edition supersedes them. |

### `business_events`

Company events that explain bursts, hubs and discontinuities in the data. Use them: several perfectly ordinary structures are only explainable with this calendar.

**Grain:** event  ·  **Rows:** 10

| Column | Description |
|---|---|
| `event_name` | Event name. |
| `start_date` | Start date. |
| `end_date` | End date where the event spans a period. |
| `markets` | Markets affected ('ALL' or a '|' separated list). |
| `description` | What the event did to the data. |

## External sources

### `registry_R/statements.json` — beneficial-ownership register (Source R)

Statement-oriented JSON in the spirit of the Beneficial Ownership Data Standard. Three record types:

- `entity` — a legal entity: name (native script), jurisdiction, identifiers (register number, LEI), whether it is publicly listed.
- `person` — a natural person: name, nationality, and a political-exposure status (`isPep` or `unknown`).
- `relationship` — an ownership or control statement: `subject` (the entity owned), `interestedParty`
  (an entity statement, a person statement, or an `unspecifiedReason` where the chain stops), and
  `interests` with the share, the control basis (shares, board control, nominee, trust) and start/end dates.

`statementDate` is when the register published the statement, which can be months after the ownership
actually changed. Coverage varies by jurisdiction (`BO_REGISTER_ACCESS` in `country_risk_context`),
so a missing path is usually missing data, not concealment. Names may differ from the vendor master
in script, transliteration or spelling, and sources occasionally disagree about the same entity.

### `registry_L_parents.csv` — corporate parent relationships (Source L)

LEI-style accounting parents, corporate only — it never reaches a natural person, and an
accounting parent is not the same thing as a beneficial owner.

| Column | Meaning |
|---|---|
| `child_lei`, `child_legal_name` | The subsidiary (join `third_parties.lei`) |
| `parent_lei`, `parent_legal_name` | The accounting parent |
| `relationship_type` | IS_DIRECTLY_CONSOLIDATED_BY |
| `relationship_start` | Start of the relationship |
| `relationship_status` | ACTIVE / INACTIVE |

## The labelled practice file

`labelled_cases_TRAINING_SET` (starter set only) lists the planted cases with their member records
and a one-line explanation. Build against it, then expect the main dataset to arrive unlabelled: it
is scored blind, and it contains patterns this practice set does not.

## Submitting results

`submission_template.csv` shows the format for ranked review cases. One row per case, most
suspicious first. Ranks must be unique and members must be `record_id` values from `third_parties`.


---
Generated by meridian-synth 0.1.0-draft · snapshot 2026-06-30 · seed 20260918
