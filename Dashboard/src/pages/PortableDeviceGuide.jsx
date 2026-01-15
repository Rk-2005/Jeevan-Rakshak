import React, { useState } from 'react';
import { useStateContext } from '../contexts/ContextProvider';
import { FaMicroscope, FaCheckCircle, FaInfoCircle, FaExclamationTriangle } from 'react-icons/fa';

const PortableDeviceGuide = () => {
  const { currentColor } = useStateContext();
  const [expandedStep, setExpandedStep] = useState(null);

  const steps = [
    {
      id: 1,
      title: 'Device Overview - Portable Water & Health Diagnostic Kit',
      icon: <FaMicroscope />,
      content: `The Portable Water & Health Diagnostic Kit is a compact, battery-operated device designed for field testing during home visits. It combines water quality analysis with basic health screening capabilities to identify potential disease outbreaks and water contamination issues.

Key Features:
• Digital pH meter
• Turbidity sensor (for water clarity)
• Bacterial count indicator
• Temperature sensor
• Data storage capability (stores up to 500 readings)
• Battery life: 8-10 hours of continuous use`,
      highlights: [
        'Compact design (fits in a bag)',
        'No internet needed for operation',
        'Battery operated (4 AAA batteries)',
        'Waterproof and durable'
      ]
    },
    {
      id: 2,
      title: 'Pre-Use Checklist & Device Preparation',
      icon: <FaCheckCircle />,
      content: `Before heading out for home visits, ensure your device is properly prepared:

1. Check Battery Level
   - Open battery compartment (bottom of device)
   - Replace batteries if indicator shows "LOW"
   - Close compartment securely

2. Calibrate Sensors
   - Turn ON device using power button
   - Wait for 30 seconds for sensors to initialize
   - Place in calibration solution (provided in kit)
   - Wait for beeping sound (approximately 1 minute)
   - Device is ready when display shows "READY"

3. Inspect Physical Condition
   - Check for cracks or damage
   - Ensure all ports are clean and dry
   - Verify carrying case is intact`,
      highlights: [
        'Check battery indicator light',
        'Calibrate before each use',
        'Wait for device initialization',
        'Inspect for physical damage'
      ]
    },
    {
      id: 3,
      title: 'Water Quality Testing Procedure',
      icon: <FaInfoCircle />,
      content: `Step-by-step guide to collect and test water samples:

1. Sample Collection
   - Use sterile test tube from kit (labeled "A")
   - Fill with water from the source being tested
   - Leave 1 inch of space from the top
   - Cap tightly and shake gently 3 times
   - Label with location and time

2. Insert Sample
   - Open sample chamber (top of device)
   - Insert test tube fully until it clicks
   - Close chamber cover gently

3. Run Analysis
   - Press "TEST" button
   - Wait for device to scan (2-3 minutes)
   - Device will beep twice when complete

4. Read Results
   - Check display screen for:
     • pH Value (normal: 6.5-8.5)
     • Turbidity (normal: <1 NTU)
     • Bacterial Count (normal: 0 CFU/ml)
     • Temperature reading

5. Record Data
   - Write results in field notebook
   - Take photo of display screen
   - Note any visual observations (color, odor, etc.)`,
      highlights: [
        'Use sterile test tubes only',
        'Fill tube properly',
        'Wait 2-3 minutes for results',
        'Document all readings'
      ]
    },
    {
      id: 4,
      title: 'Health Screening & Symptom Assessment',
      icon: <FaCheckCircle />,
      content: `Using the device for health monitoring during home visits:

1. Patient Interaction
   - Greet patient and explain the device purpose
   - Assure them it\'s non-invasive
   - Get verbal consent before testing

2. Temperature Reading
   - Use infrared thermometer (included in kit)
   - Point at forehead from 2 inches distance
   - Press button and wait for beep
   - Normal: 97.5°F to 99.5°F (36.4°C to 37.5°C)

3. Ask Health Questions
   - Inquire about recent symptoms
   - Check for common water-borne disease signs:
     • Diarrhea
     • Nausea/Vomiting
     • Abdominal pain
     • Fever
   - Document responses

4. Cross-Reference Water Quality
   - Compare water test results with symptoms
   - If poor water quality + symptoms = potential contamination
   - If symptoms only = possible viral infection

5. Recommend Actions
   - If water quality is poor: suggest boiling or filtration
   - If symptoms present: recommend medical consultation
   - If both: escalate to health authority immediately`,
      highlights: [
        'Maintain patient confidentiality',
        'Get clear consent',
        'Keep thermometer clean',
        'Document symptoms accurately'
      ]
    },
    {
      id: 5,
      title: 'Data Interpretation & Alert System',
      icon: <FaExclamationTriangle />,
      content: `Understanding the results and taking appropriate action:

Red Flags - Contact Health Authority Immediately:
✗ pH < 6.5 or > 8.5 (acidic or alkaline water)
✗ Bacterial count > 10 CFU/ml (contamination detected)
✗ Turbidity > 5 NTU (very cloudy water)
✗ Multiple patients with fever/diarrhea in same area
✗ Temperature > 101°F (38.3°C)

Yellow Flags - Monitor & Recommend Actions:
⚠ pH 6.0-6.5 or 8.5-9.0 (borderline safe)
⚠ Bacterial count 1-10 CFU/ml (some contamination)
⚠ Turbidity 1-5 NTU (slightly cloudy)
⚠ Single patient with mild symptoms

Green - Safe:
✓ pH 6.5-8.5
✓ Bacterial count 0 CFU/ml
✓ Turbidity < 1 NTU
✓ No symptoms reported`,
      highlights: [
        'Know red flag indicators',
        'Contact authorities for serious issues',
        'Document all alerts',
        'Follow up appropriately'
      ]
    },
    {
      id: 6,
      title: 'Data Management & Reporting',
      icon: <FaCheckCircle />,
      content: `How to store and report your findings:

1. In-Field Documentation
   - Record all readings in field notebook
   - Write location, date, time, name of observer
   - Take photos of water sources if available
   - Note environmental conditions (weather, sanitation)

2. Data Transfer to App
   - At end of day, submit health report through app
   - Enter: village name, symptoms, disease (if suspected)
   - Attach: photos, notes, device readings
   - System automatically sends to authorities

3. Device Memory Management
   - Device stores data internally
   - Download stored data weekly using USB cable
   - This backup ensures no data loss
   - Clear device memory after download if needed

4. Emergency Reporting
   - If critical red flags detected:
     • Immediately contact nearest health center
     • Use app "Alert" feature for urgent cases
     • Phone number on device kit for support

5. Follow-up Actions
   - Track if reported cases confirmed by health authority
   - Document follow-up visits
   - Monitor if recommended interventions were taken
   - Maintain confidentiality of patient information`,
      highlights: [
        'Keep detailed records',
        'Submit data through app',
        'Back up device memory',
        'Escalate emergencies quickly'
      ]
    },
    {
      id: 7,
      title: 'Maintenance & Troubleshooting',
      icon: <FaCheckCircle />,
      content: `Keep your device in optimal working condition:

Daily Maintenance:
1. Clean exterior with dry cloth
2. Ensure sample chamber is empty
3. Check battery indicator
4. Store in carrying case

Weekly Maintenance:
1. Clean sample chamber with distilled water
2. Wipe sensors with soft cloth
3. Check for any visible damage
4. Download stored data

Common Issues & Solutions:

Issue: "Device won't turn ON"
Solution: Replace batteries, check power button

Issue: "Test results seem incorrect"
Solution: Recalibrate device, check sample freshness

Issue: "Display shows ERROR CODE"
Solution: Restart device, contact support

Issue: "Thermometer gives inconsistent readings"
Solution: Clean lens, wait for device to settle, retry

Storage Tips:
• Keep in cool, dry place (15-25°C)
• Store away from direct sunlight
• Keep calibration solutions sealed
• Check calibration monthly`,
      highlights: [
        'Daily cleaning routine',
        'Weekly data backup',
        'Proper storage conditions',
        'Contact support when needed'
      ]
    },
    {
      id: 8,
      title: 'Safety Protocols & Best Practices',
      icon: <FaExclamationTriangle />,
      content: `Important safety guidelines for field work:

Personal Safety:
• Always carry identification
• Inform someone about your daily route
• Visit homes during daylight hours when possible
• Respect local customs and privacy
• Maintain professional appearance

Sample Handling:
• Always use fresh samples
• Never taste or touch water directly
• Use only provided sterile tubes
• Wash hands after sample collection
• Dispose of samples as instructed

Health Protection:
• Wear provided hand gloves
• Use hand sanitizer between visits
• Get vaccinated per health authority guidelines
• Report any symptoms immediately
• Maintain personal hygiene

Data Protection:
• Never share individual patient information
• Keep field notes secure
• Use password on app
• Report any data breaches
• Comply with privacy laws

Environmental Safety:
• Don't contaminate water sources
• Properly dispose of used materials
• Avoid exposure to hazardous chemicals
• Report unsafe conditions
• Use proper PPE when needed`,
      highlights: [
        'Prioritize personal safety',
        'Maintain hygiene protocols',
        'Protect patient privacy',
        'Handle data responsibly'
      ]
    }
  ];

  return (
    <div className="mt-24 px-4 md:px-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div
            className="inline-block p-4 rounded-full mb-4"
            style={{ backgroundColor: `${currentColor}20` }}
          >
            <FaMicroscope className="text-4xl" style={{ color: currentColor }} />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 dark:text-white mb-3">
            Portable Water & Health Diagnostic Kit
          </h1>
          <h2 className="text-xl text-gray-600 dark:text-gray-300 mb-4">
            Complete User Guide for ASHA Workers
          </h2>
          <p className="text-gray-500 dark:text-gray-400">
            Master the use of your portable diagnostic device for effective community health monitoring
          </p>
        </div>

        {/* Quick Tips Card */}
        <div
          className="mb-8 p-6 rounded-xl text-white"
          style={{ backgroundColor: currentColor }}
        >
          <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
            <FaInfoCircle /> Quick Tips
          </h3>
          <ul className="space-y-2 text-sm">
            <li>✓ Always calibrate before first use of the day</li>
            <li>✓ Keep device away from extreme temperatures</li>
            <li>✓ Document all readings immediately</li>
            <li>✓ Report critical findings to authorities within 24 hours</li>
          </ul>
        </div>

        {/* Steps */}
        <div className="space-y-4">
          {steps.map((step) => (
            <div
              key={step.id}
              className="bg-white dark:bg-secondary-dark-bg rounded-xl shadow-lg overflow-hidden"
            >
              {/* Step Header */}
              <button
                onClick={() => setExpandedStep(expandedStep === step.id ? null : step.id)}
                className="w-full px-6 py-4 flex items-start justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                style={{
                  borderLeft: `5px solid ${currentColor}`
                }}
              >
                <div className="flex items-start gap-4 flex-1 text-left">
                  <div
                    className="p-3 rounded-lg text-xl text-white flex-shrink-0 mt-1"
                    style={{ backgroundColor: currentColor }}
                  >
                    {step.icon}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 dark:text-white">
                      Step {step.id}: {step.title}
                    </h3>
                  </div>
                </div>
                <div className="text-2xl text-gray-400 ml-4 flex-shrink-0">
                  {expandedStep === step.id ? '−' : '+'}
                </div>
              </button>

              {/* Step Content */}
              {expandedStep === step.id && (
                <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                  <p className="text-gray-700 dark:text-gray-300 mb-4 whitespace-pre-wrap leading-relaxed">
                    {step.content}
                  </p>

                  {/* Highlights */}
                  <div className="bg-white dark:bg-secondary-dark-bg p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                    <p className="font-semibold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
                      <FaCheckCircle style={{ color: currentColor }} />
                      Key Points to Remember
                    </p>
                    <ul className="space-y-2">
                      {step.highlights.map((highlight, idx) => (
                        <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                          <span
                            className="text-lg mt-0.5 flex-shrink-0"
                            style={{ color: currentColor }}
                          >
                            •
                          </span>
                          {highlight}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-12 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
          <h3 className="font-bold text-gray-800 dark:text-white mb-3 flex items-center gap-2">
            <FaInfoCircle className="text-blue-600 dark:text-blue-400" />
            Need Help?
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-3">
            If you encounter any issues with your device or have questions:
          </p>
          <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
            <li>📞 Support Hotline: 1-800-HEALTH-01</li>
            <li>📧 Email: support@jeevanrakshak.org</li>
            <li>📱 WhatsApp: Send photos of device display for quick analysis</li>
            <li>🔗 Visit online portal for video tutorials</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PortableDeviceGuide;
