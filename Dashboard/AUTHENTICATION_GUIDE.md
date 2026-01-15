# Authentication System Guide

## Overview
The JeevanRakshak application now has a complete authentication system with role-based access control. Users can register, login, submit complaints, track their complaint status, and admins can manage all complaints.

## Features Implemented

### 1. User Authentication
- **Email/Password Login**: Users can create accounts and login using Firebase Authentication
- **Session Management**: User sessions are persisted using localStorage
- **User Data**: User information stored in Firebase Realtime Database under `users/{userId}`

### 2. User Portal (My Complaints)
- **View Own Complaints**: Users can only see complaints they've submitted
- **Status Tracking**: Real-time status updates (Pending, Reviewed, In Progress, Resolved, Closed)
- **Priority Display**: AI-detected priority levels (High, Medium, Low)
- **Admin Notes**: View notes added by admins during status updates
- **Complaint Details**: Full information including images, descriptions, and timestamps

### 3. Admin Dashboard
- **View All Complaints**: Admins can see all submitted complaints
- **Status Management**: Update complaint status with admin notes
- **Statistics**: Real-time counts of pending, in-progress, and resolved complaints
- **Bulk Management**: Sort and filter complaints by status, priority, or date

### 4. Complaint Submission
- **User Association**: Complaints automatically linked to logged-in user
- **AI Priority Detection**: Gemini API analyzes complaint to determine urgency
- **Image Upload**: Support for photo evidence via Firebase Storage
- **Authentication Required**: Users must be logged in to submit complaints

## User Flows

### For Regular Users:
1. **Register/Login** → Navigate to `/login`
2. **Submit Complaint** → Go to `/submit-complaint` (must be logged in)
3. **Track Complaints** → Visit `/my-complaints` to see status updates
4. **Logout** → Click logout button in My Complaints page

### For Admins:
1. **Login** → Navigate to `/login` (use admin account)
2. **Access Admin Dashboard** → Go to `/admin-dashboard`
3. **Manage Complaints** → Click "Update Status" on any complaint
4. **Add Notes** → Provide update details visible to users
5. **Change Status** → Select from: Pending, Reviewed, In Progress, Resolved, Closed

## Database Structure

### Users Collection (`users/{userId}`)
```javascript
{
  email: "user@example.com",
  name: "User Name",
  role: "user" | "admin",
  createdAt: timestamp
}
```

### Complaints Collection (`complaints/{complaintId}`)
```javascript
{
  id: "C1234567890",
  userId: "user_firebase_uid",
  userEmail: "user@example.com",
  name: "User Name",
  email: "user@example.com",
  phone: "+1234567890",
  location: "Location details",
  type: "Water Quality" | "Health Concern" | "Infrastructure",
  description: "Complaint description",
  userUrgency: "Low" | "Medium" | "High",
  priority: "Low" | "Medium" | "High", // AI-detected
  imageUrl: "https://...",
  status: "Pending" | "Reviewed" | "In Progress" | "Resolved" | "Closed",
  date: "2024-01-01T00:00:00.000Z",
  createdAt: 1234567890,
  adminNotes: "Admin update message" // Optional
}
```

## Routes

| Route | Component | Access | Description |
|-------|-----------|--------|-------------|
| `/login` | Login.jsx | Public | Register/Login page |
| `/submit-complaint` | ComplaintForm.jsx | Authenticated | Submit new complaint |
| `/my-complaints` | MyComplaints.jsx | Authenticated | User's complaint portal |
| `/admin-dashboard` | AdminDashboard.jsx | Admin Only* | Manage all complaints |
| `/view-complaints` | SubmittedComplaints.jsx | Public | View all complaints (read-only) |

*Note: Currently admin access is not enforced at route level. Consider adding ProtectedRoute component for production.

## Navigation

### Sidebar Links (dummy.js):
- **Community Section**:
  - Submit Complaint
  - My Complaints (user portal)
  - View Complaints (public view)
  - Complaints Awareness
  
- **Administration Section**:
  - Admin Dashboard (complaint management)
  - Login (authentication)

## Status Values

### Complaint Statuses:
1. **Pending** - New complaint, awaiting review
2. **Reviewed** - Admin has seen the complaint
3. **In Progress** - Work is being done
4. **Resolved** - Issue has been fixed
5. **Closed** - Complaint is closed (no further action)

### Priority Levels (AI-Detected):
1. **High** - Urgent, requires immediate attention
2. **Medium** - Important, normal processing
3. **Low** - Non-urgent, can be scheduled

## Security Considerations

### Current Implementation:
- ✅ Email/password authentication via Firebase Auth
- ✅ User session management with localStorage
- ✅ User-specific complaint filtering
- ✅ Admin notes and status updates

### Recommended Improvements for Production:
- ⚠️ Add route guards to protect authenticated pages
- ⚠️ Implement role-based access control middleware
- ⚠️ Add email verification requirement
- ⚠️ Implement password reset functionality
- ⚠️ Add rate limiting for API calls (Gemini API)
- ⚠️ Validate user role on backend before status updates
- ⚠️ Add CSRF protection
- ⚠️ Implement secure admin invite system

## Testing the System

### Create Admin Account:
1. Register a new user via `/login`
2. Manually update the user's role in Firebase Console:
   - Go to Firebase Console → Realtime Database
   - Navigate to `users/{userId}`
   - Change `role` from `"user"` to `"admin"`
3. Logout and login again
4. Access `/admin-dashboard`

### Test User Flow:
1. Register as a new user
2. Submit a complaint with image
3. Check AI-detected priority
4. Navigate to My Complaints
5. Verify complaint appears

### Test Admin Flow:
1. Login as admin user
2. Go to Admin Dashboard
3. Click "Update Status" on a complaint
4. Add notes and change status
5. Verify status updates in user's portal

## Components

### Authentication Components:
- **Login.jsx** - Login/Signup form with Firebase Auth
- **MyComplaints.jsx** - User portal for tracking complaints
- **AdminDashboard.jsx** - Admin interface for managing complaints

### Styling Files:
- **Auth.css** - Login/Signup page styles
- **MyComplaints.css** - User portal styles
- **AdminDashboard.css** - Admin dashboard styles

### Updated Components:
- **ComplaintForm.jsx** - Now requires login and associates with user
- **App.js** - Added authentication routes
- **pages/index.jsx** - Export new components
- **data/dummy.js** - Updated sidebar navigation

## Firebase Configuration

Ensure your `firebaseConfig.js` has:
```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "jalrakshak-dc076.firebaseapp.com",
  projectId: "jalrakshak-dc076",
  storageBucket: "jalrakshak-dc076.firebasestorage.app",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID",
  measurementId: "YOUR_MEASUREMENT_ID",
  databaseURL: "https://jalrakshak-dc076-default-rtdb.firebaseio.com/"
};
```

## Troubleshooting

### "Please login to submit a complaint" Error:
- User is not logged in
- localStorage doesn't have userId
- Solution: Navigate to `/login` and sign in

### Complaints Not Showing in My Complaints:
- Check userId in localStorage matches complaint userId
- Verify complaints in Firebase have userId field
- Old complaints may not have userId (submit new ones)

### Admin Dashboard Not Accessible:
- Verify user role is "admin" in Firebase Console
- Check localStorage has userRole set to "admin"
- Clear localStorage and login again

### Status Updates Not Syncing:
- Check Firebase Realtime Database rules allow updates
- Verify internet connection
- Check browser console for errors

## Next Steps

### Recommended Enhancements:
1. **Password Reset**: Add forgot password functionality
2. **Email Verification**: Require email verification on signup
3. **Profile Management**: Allow users to update their profile
4. **Notifications**: Add real-time notifications for status changes
5. **Admin Roles**: Create super-admin and moderator roles
6. **Complaint Assignment**: Allow admins to assign complaints to specific staff
7. **Analytics Dashboard**: Add charts showing complaint trends
8. **Export Reports**: Allow admins to export complaint reports
9. **Search Functionality**: Add search and advanced filters
10. **Mobile Responsiveness**: Optimize layouts for mobile devices

## Support

For issues or questions:
- Check Firebase Console for database/auth errors
- Review browser console for JavaScript errors
- Verify all dependencies are installed: `npm install`
- Ensure Firebase project has Authentication, Database, and Storage enabled
