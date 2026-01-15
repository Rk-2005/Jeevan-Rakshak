// Test Data Generator for Complaint System
// This file contains sample complaints to test the AI priority detection

export const sampleComplaints = [
  // HIGH PRIORITY COMPLAINTS
  {
    name: "Dr. Rajesh Kumar",
    email: "rajesh.kumar@example.com",
    phone: "+91 98765 43210",
    location: "Village Rampur, District Kamrup, Assam",
    complaintType: "Health Alert",
    urgency: "Critical",
    description: "URGENT: Outbreak of severe diarrhea in our village. 15 children and 8 adults hospitalized in last 48 hours. All cases linked to community tube well near school. Water appears discolored. Multiple families affected. Immediate intervention required.",
  },
  {
    name: "Priya Sharma",
    email: "priya.s@example.com",
    phone: "+91 98765 43211",
    location: "Block 5, Shillong, Meghalaya",
    complaintType: "Water Quality Issue",
    urgency: "Critical",
    description: "Suspected cholera outbreak. 3 confirmed cases, 10 suspected. All patients consumed water from the main supply line. Community of 200+ families at risk. Water testing needed urgently.",
  },
  {
    name: "Amit Patel",
    email: "amit.patel@example.com",
    phone: "+91 98765 43212",
    location: "Remote Hamlet, Kohima District, Nagaland",
    complaintType: "Health Alert",
    urgency: "Critical",
    description: "Typhoid epidemic suspected. 5 children showing symptoms - high fever, abdominal pain, vomiting. Village cut off by floods, only water source is contaminated well. Medical supplies running low. Require emergency water purification support.",
  },

  // MEDIUM PRIORITY COMPLAINTS
  {
    name: "Sunita Devi",
    email: "sunita.devi@example.com",
    phone: "+91 98765 43213",
    location: "Village Moirang, Manipur",
    complaintType: "Water Quality Issue",
    urgency: "Moderate",
    description: "Community hand pump water has foul smell and yellowish tint for past week. Lab test shows high turbidity and bacterial presence. 50 families dependent on this source. No illness reported yet but concerned about health risks.",
  },
  {
    name: "Karan Singh",
    email: "karan.singh@example.com",
    phone: "+91 98765 43214",
    location: "Township Area, Aizawl, Mizoram",
    complaintType: "Infrastructure Problem",
    urgency: "Moderate",
    description: "Main water pipeline leaking severely, causing water pressure issues for entire neighborhood. 100+ households affected. Water supply irregular and contamination risk from sewage line proximity.",
  },
  {
    name: "Lakshmi Nair",
    email: "lakshmi.n@example.com",
    phone: "+91 98765 43215",
    location: "Block C, Imphal, Manipur",
    complaintType: "Contamination Report",
    urgency: "Moderate",
    description: "Industrial waste suspected in water supply. Water has chemical odor and metallic taste. Several complaints from neighbors. Need water quality assessment and source investigation.",
  },

  // LOW PRIORITY COMPLAINTS
  {
    name: "Rohit Verma",
    email: "rohit.verma@example.com",
    phone: "+91 98765 43216",
    location: "Urban Area, Guwahati, Assam",
    complaintType: "Infrastructure Problem",
    urgency: "Low",
    description: "Hand pump handle loose and making noise. Still functional but needs maintenance. Non-urgent repair request for community well.",
  },
  {
    name: "Anjali Das",
    email: "anjali.das@example.com",
    phone: "+91 98765 43217",
    location: "Residential Colony, Shillong",
    complaintType: "Other",
    urgency: "Low",
    description: "Water tank exterior needs repainting. Rust visible on outside surface but not affecting water quality. Cosmetic maintenance required.",
  },
  {
    name: "Vikram Joshi",
    email: "vikram.j@example.com",
    phone: "+91 98765 43218",
    location: "Village Panchayat Office, Tripura",
    complaintType: "Supply Disruption",
    urgency: "Low",
    description: "Water supply timing inconsistent. Usually comes at 6 AM, sometimes delayed to 8 AM. Not a major issue but prefer more regular schedule.",
  },
];

// Instructions for using this test data:
/*
1. Copy any of the sample complaint objects above
2. Go to http://localhost:3000/submit-complaint
3. Paste the values into the form fields
4. Submit and observe the AI priority detection
5. Check if the detected priority matches the expected category

Expected Results:
- First 3 complaints → HIGH priority
- Middle 3 complaints → MEDIUM priority  
- Last 3 complaints → LOW priority

This helps validate the Gemini AI analysis accuracy!
*/

// Quick test function (optional - for developers)
export const testComplaintPriority = async (complaint, geminiApiKey) => {
  const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${geminiApiKey}`;
  
  const prompt = `Analyze the following water/health complaint and determine its priority level (HIGH, MEDIUM, or LOW).

Complaint Details:
- Type: ${complaint.complaintType}
- Location: ${complaint.location}
- Description: ${complaint.description}
- User Indicated Urgency: ${complaint.urgency}

Priority Classification Guidelines:
- HIGH: Immediate health risks, disease outbreaks, severe contamination, multiple people affected
- MEDIUM: Water quality concerns, infrastructure issues affecting supply, potential health risks
- LOW: Minor issues, cosmetic concerns, non-urgent maintenance requests

Respond with ONLY ONE WORD: HIGH, MEDIUM, or LOW`;

  try {
    const response = await fetch(GEMINI_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
      }),
    });

    const data = await response.json();
    const priorityText = data.candidates[0].content.parts[0].text.trim().toUpperCase();
    
    if (priorityText.includes('HIGH')) return 'High';
    if (priorityText.includes('MEDIUM')) return 'Medium';
    if (priorityText.includes('LOW')) return 'Low';
    
    return 'Medium';
  } catch (error) {
    console.error('Error testing priority:', error);
    return 'Error';
  }
};
