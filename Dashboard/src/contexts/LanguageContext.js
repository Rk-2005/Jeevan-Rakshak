import React, { createContext, useContext, useState } from 'react';

const LanguageContext = createContext();

const translations = {
  en: {
    // Navigation
    dashboard: 'Dashboard',
    waterQuality: 'Water Quality',
    healthData: 'Health Data',
    outbreakRisk: 'Outbreak Risk',
    complaints: 'Complaints',
    awareness: 'Awareness',
    settings: 'Settings',

    // Dashboard
    waterSaved: 'Water Saved',
    unsafeWaterSources: 'Unsafe Water Sources',
    detectedOutbreakRisks: 'Detected Outbreak Risks',
    reportedComplaints: 'Reported Complaints',
    activeSensors: 'Active Sensors',
    downloadReport: 'Download Report',
    outbreaksDetections: 'Outbreaks Detections',
    monthlyStatistics: 'Monthly statistics',
    monthlyDetects: 'Monthly Detects',
    downloadApp: 'Download App',
    appDescription: 'A go-to solution for all your water management problems. Get real-time alerts and reports directly on your phone.',

    // Water Quality
    waterQualityPrediction: 'Water Quality Prediction',
    chooseCsvFile: 'Choose CSV File',
    predictWaterQuality: 'Predict Water Quality',
    predictionResults: 'Prediction Results',
    sample: 'Sample',
    potability: 'Potability',

    // Health Data Collection
    healthDataCollection: 'Health Data Collection',
    villageName: 'Village Name',
    patientAgeGroup: 'Patient Age Group',
    reportedSymptoms: 'Reported Symptoms',
    confirmedDisease: 'Confirmed Disease (Optional)',
    submitReport: 'Submit Report',

    // Outbreak Risk
    outbreakRiskLevel: 'Outbreak Risk Level',
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    riskScore: 'Risk Score',
    hotspots: 'Hotspots',

    // Alerts
    alerts: 'Alerts',
    highOutbreakRisk: 'High outbreak risk detected in',
    unsafeWaterSource: 'Unsafe water source found',

    // Water Quality Parameters
    ph: 'pH',
    turbidity: 'Turbidity',
    bacterialCount: 'Bacterial Count',
    chlorineLevels: 'Chlorine Levels',
    temperature: 'Temperature',
    dissolvedOxygen: 'Dissolved Oxygen',

    // Complaints
    submitComplaint: 'Submit Complaint',
    complaintType: 'Complaint Type',
    description: 'Description',
    location: 'Location',
    priority: 'Priority',

    // Awareness
    hygieneEducation: 'Hygiene Education',
    safeWaterPractices: 'Safe Water Practices',

    // Common
    submit: 'Submit',
    cancel: 'Cancel',
    save: 'Save',
    edit: 'Edit',
    delete: 'Delete',
    view: 'View',
    filter: 'Filter',
    export: 'Export',
    search: 'Search',
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    warning: 'Warning',
    info: 'Info',

    // Dashboard Additional
    welcomeAdmin: 'Welcome, Admin!',
    welcomeAshaWorker: 'Welcome, ASHA Worker!',
    welcomeUser: 'Welcome, User!',
    administrator: 'Administrator',
    ashaWorker: 'ASHA Worker',
    user: 'User',
    currentDate: 'Current Date',
    communityHealthMonitoringDashboard: 'Community Health Monitoring Dashboard',
    realtimeMonitoring: 'Real-time monitoring and analysis of water quality and health data',
    allVillages: 'All Villages',
    villageA: 'Village A',
    villageB: 'Village B',
    villageC: 'Village C',
    villageD: 'Village D',
    villageE: 'Village E',
    allDistricts: 'All Districts',
    district1: 'District 1',
    district2: 'District 2',
    district3: 'District 3',
    last7days: 'Last 7 days',
    last30days: 'Last 30 days',
    last90days: 'Last 90 days',
    lastYear: 'Last year',
    liters: 'liters',
    waterQualityTrend: 'Water Quality Trend',
    complaintStatusDistribution: 'Complaint Status Distribution',
    villagePerformanceOverview: 'Village Performance Overview',
    healthCases: 'Health Cases',
    waterQuality: 'Water Quality',
    riskLevel: 'Risk Level',
    status: 'Status',
    good: 'Good',
    fair: 'Fair',
    poor: 'Poor',
    safe: 'Safe',
    monitor: 'Monitor',
    alert: 'Alert',
    resolved: 'Resolved',
    inProgress: 'In Progress',
    pending: 'Pending',
    exportAsPDF: 'Export as PDF',
    exportAsExcel: 'Export as Excel',
    downloadReportText: 'Coming soon',
    comingSoon: 'Coming soon',
    liveDetections: 'Live Detections are on! For help with any technical issue, please contact us at 8265096155',
    dashboardMenu: 'DASHBOARD',
    healthMonitoringMenu: 'HEALTH MONITORING',
    gisMapping: 'GIS-MAPPING',
    administrationMenu: 'ADMINISTRATION',
    appsMenu: 'APPS',
  },
  hi: {
    // Navigation
    dashboard: 'डैशबोर्ड',
    waterQuality: 'जल गुणवत्ता',
    healthData: 'स्वास्थ्य डेटा',
    outbreakRisk: 'प्रकोप जोखिम',
    complaints: 'शिकायतें',
    awareness: 'जागरूकता',
    settings: 'सेटिंग्स',

    // Dashboard
    waterSaved: 'बचाया गया पानी',
    unsafeWaterSources: 'असुरक्षित जल स्रोत',
    detectedOutbreakRisks: 'पता चला प्रकोप जोखिम',
    reportedComplaints: 'रिपोर्ट की गई शिकायतें',
    activeSensors: 'सक्रिय सेंसर',
    downloadReport: 'रिपोर्ट डाउनलोड करें',
    outbreaksDetections: 'प्रकोप पहचान',
    monthlyStatistics: 'मासिक आंकड़े',
    monthlyDetects: 'मासिक पहचान',
    downloadApp: 'ऐप डाउनलोड करें',
    appDescription: 'आपकी सभी जल प्रबंधन समस्याओं के लिए एक समाधान। अपने फोन पर सीधे रियल-टाइम अलर्ट और रिपोर्ट प्राप्त करें।',

    // Water Quality
    waterQualityPrediction: 'जल गुणवत्ता भविष्यवाणी',
    chooseCsvFile: 'CSV फ़ाइल चुनें',
    predictWaterQuality: 'जल गुणवत्ता का अनुमान लगाएं',
    predictionResults: 'भविष्यवाणी परिणाम',
    sample: 'नमूना',
    potability: 'पीने योग्यता',

    // Health Data Collection
    healthDataCollection: 'स्वास्थ्य डेटा संग्रह',
    villageName: 'गांव का नाम',
    patientAgeGroup: 'रोगी आयु समूह',
    reportedSymptoms: 'रिपोर्ट किए गए लक्षण',
    confirmedDisease: 'पुष्टि की गई बीमारी (वैकल्पिक)',
    submitReport: 'रिपोर्ट सबमिट करें',

    // Outbreak Risk
    outbreakRiskLevel: 'प्रकोप जोखिम स्तर',
    low: 'कम',
    medium: 'मध्यम',
    high: 'उच्च',
    riskScore: 'जोखिम स्कोर',
    hotspots: 'हॉटस्पॉट',

    // Alerts
    alerts: 'अलर्ट',
    highOutbreakRisk: 'उच्च प्रकोप जोखिम का पता चला',
    unsafeWaterSource: 'असुरक्षित जल स्रोत मिला',

    // Water Quality Parameters
    ph: 'pH',
    turbidity: 'टर्बिडिटी',
    bacterialCount: 'बैक्टीरियल काउंट',
    chlorineLevels: 'क्लोरीन स्तर',
    temperature: 'तापमान',
    dissolvedOxygen: 'घुलित ऑक्सीजन',

    // Complaints
    submitComplaint: 'शिकायत सबमिट करें',
    complaintType: 'शिकायत प्रकार',
    description: 'विवरण',
    location: 'स्थान',
    priority: 'प्राथमिकता',

    // Awareness
    hygieneEducation: 'स्वच्छता शिक्षा',
    safeWaterPractices: 'सुरक्षित जल प्रथाएं',

    // Common
    submit: 'सबमिट करें',
    cancel: 'रद्द करें',
    save: 'सहेजें',
    edit: 'संपादित करें',
    delete: 'हटाएं',
    view: 'देखें',
    filter: 'फ़िल्टर',
    export: 'निर्यात',
    search: 'खोजें',
    loading: 'लोड हो रहा है...',
    error: 'त्रुटि',
    success: 'सफलता',
    warning: 'चेतावनी',
    info: 'जानकारी',

    // Dashboard Additional
    welcomeAdmin: 'स्वागत है, व्यवस्थापक!',
    welcomeAshaWorker: 'स्वागत है, आशा कार्यकर्ता!',
    welcomeUser: 'स्वागत है, उपयोगकर्ता!',
    administrator: 'व्यवस्थापक',
    ashaWorker: 'आशा कार्यकर्ता',
    user: 'उपयोगकर्ता',
    currentDate: 'वर्तमान तारीख',
    communityHealthMonitoringDashboard: 'समुदाय स्वास्थ्य निगरानी डैशबोर्ड',
    realtimeMonitoring: 'जल गुणवत्ता और स्वास्थ्य डेटा की रीयल-टाइम निगरानी और विश्लेषण',
    allVillages: 'सभी गांव',
    villageA: 'गांव A',
    villageB: 'गांव B',
    villageC: 'गांव C',
    villageD: 'गांव D',
    villageE: 'गांव E',
    allDistricts: 'सभी जिले',
    district1: 'जिला 1',
    district2: 'जिला 2',
    district3: 'जिला 3',
    last7days: 'पिछले 7 दिन',
    last30days: 'पिछले 30 दिन',
    last90days: 'पिछले 90 दिन',
    lastYear: 'पिछला वर्ष',
    liters: 'लीटर',
    waterQualityTrend: 'जल गुणवत्ता प्रवृत्ति',
    complaintStatusDistribution: 'शिकायत स्थिति वितरण',
    villagePerformanceOverview: 'गांव प्रदर्शन सारांश',
    healthCases: 'स्वास्थ्य मामले',
    waterQuality: 'जल गुणवत्ता',
    riskLevel: 'जोखिम स्तर',
    status: 'स्थिति',
    good: 'अच्छा',
    fair: 'ठीक है',
    poor: 'खराब',
    safe: 'सुरक्षित',
    monitor: 'निगरानी',
    alert: 'अलर्ट',
    resolved: 'समाधान किया गया',
    inProgress: 'प्रगति में',
    pending: 'लंबित',
    exportAsPDF: 'पीडीएफ के रूप में निर्यात करें',
    exportAsExcel: 'एक्सल के रूप में निर्यात करें',
    downloadReportText: 'जल्द आ रहा है',
    comingSoon: 'जल्द आ रहा है',
    liveDetections: 'लाइव डिटेक्शन चालू है! किसी भी तकनीकी समस्या के लिए, कृपया हमसे 8265096155 पर संपर्क करें',
    dashboardMenu: 'डैशबोर्ड',
    healthMonitoringMenu: 'स्वास्थ्य निगरानी',
    gisMapping: 'जीआईएस-मैपिंग',
    administrationMenu: 'प्रशासन',
    appsMenu: 'ऐप्स',
  },
  bn: {
    // Navigation
    dashboard: 'ড্যাশবোর্ড',
    waterQuality: 'জল গুণমান',
    healthData: 'স্বাস্থ্য ডেটা',
    outbreakRisk: 'প্রাদুর্ভাব ঝুঁকি',
    complaints: 'অভিযোগ',
    awareness: 'সচেতনতা',
    settings: 'সেটিংস',

    // Dashboard
    waterSaved: 'সংরক্ষিত জল',
    unsafeWaterSources: 'অনিরাপদ জল উৎস',
    detectedOutbreakRisks: 'সনাক্তকৃত প্রাদুর্ভাব ঝুঁকি',
    reportedComplaints: 'রিপোর্ট করা অভিযোগ',
    activeSensors: 'সক্রিয় সেন্সর',
    downloadReport: 'রিপোর্ট ডাউনলোড করুন',
    outbreaksDetections: 'প্রাদুর্ভাব সনাক্তকরণ',
    monthlyStatistics: 'মাসিক পরিসংখ্যান',
    monthlyDetects: 'মাসিক সনাক্তকরণ',
    downloadApp: 'অ্যাপ ডাউনলোড করুন',
    appDescription: 'আপনার সমস্ত জল ব্যবস্থাপনা সমস্যার জন্য একটি সমাধান। আপনার ফোনে সরাসরি রিয়েল-টাইম অ্যালার্ট এবং রিপোর্ট পান।',

    // Water Quality
    waterQualityPrediction: 'জল গুণমান ভবিষ্যদ্বাণী',
    chooseCsvFile: 'CSV ফাইল নির্বাচন করুন',
    predictWaterQuality: 'জল গুণমান ভবিষ্যদ্বাণী করুন',
    predictionResults: 'ভবিষ্যদ্বাণী ফলাফল',
    sample: 'নমুনা',
    potability: 'পানযোগ্যতা',

    // Health Data Collection
    healthDataCollection: 'স্বাস্থ্য ডেটা সংগ্রহ',
    villageName: 'গ্রামের নাম',
    patientAgeGroup: 'রোগীর বয়স গ্রুপ',
    reportedSymptoms: 'রিপোর্ট করা লক্ষণ',
    confirmedDisease: 'নিশ্চিত রোগ (ঐচ্ছিক)',
    submitReport: 'রিপোর্ট জমা দিন',

    // Outbreak Risk
    outbreakRiskLevel: 'প্রাদুর্ভাব ঝুঁকি স্তর',
    low: 'নিম্ন',
    medium: 'মাঝারি',
    high: 'উচ্চ',
    riskScore: 'ঝুঁকি স্কোর',
    hotspots: 'হটস্পট',

    // Alerts
    alerts: 'অ্যালার্ট',
    highOutbreakRisk: 'উচ্চ প্রাদুর্ভাব ঝুঁকি সনাক্ত',
    unsafeWaterSource: 'অনিরাপদ জল উৎস পাওয়া গেছে',

    // Water Quality Parameters
    ph: 'pH',
    turbidity: 'ঘোলাটে',
    bacterialCount: 'ব্যাকটেরিয়াল কাউন্ট',
    chlorineLevels: 'ক্লোরিন স্তর',
    temperature: 'তাপমাত্রা',
    dissolvedOxygen: 'দ্রবীভূত অক্সিজেন',

    // Complaints
    submitComplaint: 'অভিযোগ জমা দিন',
    complaintType: 'অভিযোগের ধরন',
    description: 'বিবরণ',
    location: 'অবস্থান',
    priority: 'অগ্রাধিকার',

    // Awareness
    hygieneEducation: 'স্বাস্থ্যবিধি শিক্ষা',
    safeWaterPractices: 'নিরাপদ জল অনুশীলন',

    // Common
    submit: 'জমা দিন',
    cancel: 'বাতিল',
    save: 'সংরক্ষণ',
    edit: 'সম্পাদনা',
    delete: 'মুছে ফেলুন',
    view: 'দেখুন',
    filter: 'ফিল্টার',
    export: 'রপ্তানি',
    search: 'অনুসন্ধান',
    loading: 'লোড হচ্ছে...',
    error: 'ত্রুটি',
    success: 'সফলতা',
    warning: 'সতর্কতা',
    info: 'তথ্য',
  },
  mr: {
    // Navigation
    dashboard: 'डॅशबोर्ड',
    waterQuality: 'पाणी गुणवत्ता',
    healthData: 'आरोग्य डेटा',
    outbreakRisk: 'रोगाचा प्रादुर्भाव धोका',
    complaints: 'तक्रारी',
    awareness: 'जागृती',
    settings: 'सेटिंग्ज',

    // Dashboard
    waterSaved: 'वाचवलेले पाणी',
    unsafeWaterSources: 'असुरक्षित पाणी स्रोत',
    detectedOutbreakRisks: 'आढळलेले प्रादुर्भाव धोके',
    reportedComplaints: 'अहवालित तक्रारी',
    activeSensors: 'सक्रिय सेंसर',
    downloadReport: 'अहवाल डाउनलोड करा',
    outbreaksDetections: 'प्रादुर्भाव शोध',
    monthlyStatistics: 'मासिक आकडेवारी',
    monthlyDetects: 'मासिक शोध',
    downloadApp: 'अॅप डाउनलोड करा',
    appDescription: 'तुमच्या सर्व पाणी व्यवस्थापन समस्यांसाठी एक उपाय. तुमच्या फोनवर थेट रिअल-टाइम अलर्ट आणि अहवाल मिळवा.',

    // Water Quality
    waterQualityPrediction: 'पाणी गुणवत्ता भविष्यकथन',
    chooseCsvFile: 'CSV फाइल निवडा',
    predictWaterQuality: 'पाणी गुणवत्ता अंदाज लावा',
    predictionResults: 'भविष्यकथन परिणाम',
    sample: 'नमुना',
    potability: 'पिण्यायोग्यता',

    // Health Data Collection
    healthDataCollection: 'आरोग्य डेटा संग्रह',
    villageName: 'गावाचे नाव',
    patientAgeGroup: 'रुग्ण वयोगट',
    reportedSymptoms: 'अहवालित लक्षणे',
    confirmedDisease: 'पुष्टीकृत रोग (पर्यायी)',
    submitReport: 'अहवाल सबमिट करा',

    // Outbreak Risk
    outbreakRiskLevel: 'प्रादुर्भाव धोका स्तर',
    low: 'कमी',
    medium: 'मध्यम',
    high: 'उच्च',
    riskScore: 'धोका स्कोअर',
    hotspots: 'हॉटस्पॉट',

    // Alerts
    alerts: 'अलर्ट',
    highOutbreakRisk: 'उच्च प्रादुर्भाव धोका आढळला',
    unsafeWaterSource: 'असुरक्षित पाणी स्रोत सापडला',

    // Water Quality Parameters
    ph: 'pH',
    turbidity: 'गढूळपणा',
    bacterialCount: 'बॅक्टेरियल काउंट',
    chlorineLevels: 'क्लोरीन स्तर',
    temperature: 'तापमान',
    dissolvedOxygen: 'विद्रवित ऑक्सिजन',

    // Complaints
    submitComplaint: 'तक्रार सबमिट करा',
    complaintType: 'तक्रार प्रकार',
    description: 'वर्णन',
    location: 'स्थान',
    priority: 'प्राधान्य',

    // Awareness
    hygieneEducation: 'स्वच्छता शिक्षण',
    safeWaterPractices: 'सुरक्षित पाणी पद्धती',

    // Common
    submit: 'सबमिट करा',
    cancel: 'रद्द करा',
    save: 'जतन करा',
    edit: 'संपादन',
    delete: 'हटवा',
    view: 'पहा',
    filter: 'फिल्टर',
    export: 'निर्यात',
    search: 'शोधा',
    loading: 'लोड होत आहे...',
    error: 'त्रुटी',
    success: 'यश',
    warning: 'चेतावणी',
    info: 'माहिती',
  },
};

export const LanguageProvider = ({ children }) => {
  const [currentLanguage, setCurrentLanguage] = useState(() => localStorage.getItem('selectedLanguage') || 'en');

  const changeLanguage = (language) => {
    setCurrentLanguage(language);
    localStorage.setItem('selectedLanguage', language);
  };

  const t = (key) => translations[currentLanguage][key] || key;

  return (
    <LanguageContext.Provider value={{ currentLanguage, changeLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
