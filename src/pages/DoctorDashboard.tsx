import React, { useState } from "react";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { 
  Users, 
  Calendar, 
  BarChart, 
  Settings, 
  MessageSquare, 
  LogOut,
  Bell,
  Video,
  Upload,
  FileText,
  AlertTriangle,
  Stethoscope
} from "lucide-react";
import { Loader2 } from "lucide-react";
import { useEffect } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import DoctorProfile from "@/components/doctor/DoctorProfile";
import ApiService from "../services/api.js";
import { useLanguage } from "@/context/LanguageContext";
//import multilingualTranslate from "../utils/translation.ts";

const name = localStorage.getItem("name");

const DoctorDashboard = () => {
  const { translate } = useLanguage();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("patients");
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [showPrescriptionDialog, setShowPrescriptionDialog] = useState(false);
  const [prescriptionImage, setPrescriptionImage] = useState<File | null>(null);
  const [prescriptionNotes, setPrescriptionNotes] = useState("");
  const [currentPatientId, setCurrentPatientId] = useState<string | null>(null);
  const [showNewAppointmentDialog, setShowNewAppointmentDialog] = useState(false);
  const [appointmentData, setAppointmentData] = useState({
    patientName: "",
    date: "",
    time: "",
    type: "Video",
  });
  // Replace your mock data with these state variables
  const [patients, setPatients] = useState([]);
  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [doctorProfile, setDoctorProfile] = useState({
    name: "",
    specialization: "",
    totalPatients: 0
  });
  const [consultations, setConsultations] = useState([]);
  // Add this useEffect to fetch data when the component mounts
  useEffect(() => {
    const fetchDoctorData = async () => {
      try {
        setIsLoading(true);
        
        // 1. Fetch doctor profile
        const profileResponse = await ApiService.getDoctorProfile();
        if (profileResponse.success && profileResponse.doctor) {
          setDoctorProfile({
            name: profileResponse.doctor.name,
            specialization: profileResponse.doctor.specialization,
            totalPatients: profileResponse.doctor.totalPatients || 0
          });
        }
        
        // 2. Fetch appointments
        const appointmentsResponse = await ApiService.getDoctorAppointments();
        if (appointmentsResponse.success && appointmentsResponse.appointments) {
          setAppointments(appointmentsResponse.appointments.map(apt => ({
            id: apt._id,
            patient: apt.patientId.name,
            time: apt.timeSlot,
            date: new Date(apt.appointmentDate).toLocaleDateString(),
            type: apt.consultationType || "Video",
            status: apt.status,
            patientId: apt.patientId._id,
            symptoms: apt.symptoms || ""
          })));

          // Add this line to set consultations based on appointments
          setConsultations(appointmentsResponse.appointments.map(apt => ({
            id: apt._id,
            patient: apt.patientId.name,
            time: apt.timeSlot,
            date: new Date(apt.appointmentDate).toLocaleDateString(),
            type: apt.consultationType || "Video",
            status: apt.status
          })));
          
          // 3. Extract unique patients from appointments
          const uniquePatients = [...new Map(
            appointmentsResponse.appointments.map(apt => [
              apt.patientId._id, 
              {
                id: apt.patientId._id,
                name: apt.patientId.name,
                age: apt.patientId.age || 30,
                gender: apt.patientId.gender || "Unknown",
                symptoms: apt.symptoms ? [apt.symptoms] : ["Consultation"],
                status: apt.status,
                lastVisit: new Date(apt.appointmentDate).toLocaleDateString(),
                nextAppointment: `${new Date(apt.appointmentDate).toLocaleDateString()}, ${apt.timeSlot}`,
                image: apt.patientId.profileImage || "/placeholder.svg"
              }
            ])
          ).values()];
          
          setPatients(uniquePatients);
        }
      } catch (error) {
        console.error("Error fetching doctor data:", error);
        toast({
          title: "Error",
          description: "Failed to load dashboard data",
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchDoctorData();
  }, []);

  const handleLogout = async () => {
  try {
    const response = await ApiService.logout();
    
    if (response.success) {
      localStorage.removeItem('userId');
      localStorage.removeItem('userName');
      localStorage.removeItem('userRole');
      
      toast({
        title: translate("logout"),
        description: translate("You have been successfully logged out.")
      });
      
      navigate("/");
    } else {
      throw new Error("Logout failed");
    }
  } catch (error) {
    console.error("Logout error:", error);
    toast({
      title: translate("logoutFailed"),
      description: translate("There was a problem logging out. Please try again."),
      variant: "destructive"
    });
  }
};

  // Update the handleStartConsultation function
  const handleStartConsultation = (consultationId: string) => {
    // Find the consultation to get details
    const consultation = consultations.find(c => c.id === consultationId);
    
    if (consultation) {
      toast({
        title: "Starting consultation",
        description: "Connecting to video call...",
      });
      
      // Navigate to the video conference with the appointment ID
      navigate(`/video-conference/${consultationId}`);
    } else {
      toast({
        title: "Error",
        description: "Consultation not found",
        variant: "destructive"
      });
    }
  };

  const handleUploadPrescription = (patientId: string) => {
    setCurrentPatientId(patientId);
    setShowPrescriptionDialog(true);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPrescriptionImage(e.target.files[0]);
    }
  };

  const handlePrescriptionSubmit = () => {
    if (!prescriptionImage) {
      toast({
        title: "Missing prescription",
        description: "Please upload the prescription image.",
        variant: "destructive"
      });
      return;
    }

    toast({
      title: "Prescription uploaded",
      description: "The prescription has been sent to the patient."
    });

    setShowPrescriptionDialog(false);
    setPrescriptionImage(null);
    setPrescriptionNotes("");
    setCurrentPatientId(null);
  };

  const handleScheduleNew = () => {
    setShowNewAppointmentDialog(true);
  };

  const handleAppointmentSubmit = () => {
    if (!appointmentData.patientName || !appointmentData.date || !appointmentData.time) {
      toast({
        title: "Missing information",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    toast({
      title: "Appointment scheduled",
      description: `New appointment with ${appointmentData.patientName} has been scheduled.`
    });

    setShowNewAppointmentDialog(false);
    setAppointmentData({
      patientName: "",
      date: "",
      time: "",
      type: "Video",
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4">
        <div className="container mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center">
            <Stethoscope className="h-8 w-8 text-health-600 mr-3" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">Dr. {name}</h1>
              <p className="text-sm text-gray-500">General Physician • IMA Certified</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="h-5 w-5" />
              <span className="absolute top-0 right-0 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
            </Button>
            
            <div className="flex items-center">
              <Avatar className="h-9 w-9 mr-2">
                <AvatarImage src="/placeholder.svg" alt="Doctor" />
                <AvatarFallback>DR</AvatarFallback>
              </Avatar>
              <div className="hidden md:block">
                <p className="text-sm font-medium">Dr. {name}</p>
                <p className="text-xs text-gray-500">IMA #MH12345</p>
              </div>
            </div>
            
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="border-b mb-6">
            <TabsList className="overflow-x-auto">
              <TabsTrigger value="dashboard">
                <BarChart className="h-4 w-4 mr-2" />
                Dashboard
              </TabsTrigger>
              <TabsTrigger value="patients">
                <Users className="h-4 w-4 mr-2" />
                Patients
              </TabsTrigger>
              <TabsTrigger value="consultations">
                <Video className="h-4 w-4 mr-2" />
                Consultations
              </TabsTrigger>
              <TabsTrigger value="prescriptions">
                <FileText className="h-4 w-4 mr-2" />
                Prescriptions
              </TabsTrigger>
              <TabsTrigger value="appointments">
                <Calendar className="h-4 w-4 mr-2" />
                Appointments
              </TabsTrigger>
              <TabsTrigger value="messages">
                <MessageSquare className="h-4 w-4 mr-2" />
                Messages
              </TabsTrigger>
              <TabsTrigger value="settings">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="dashboard" className="mt-0">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Total Patients</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">{doctorProfile.totalPatients || patients.length}</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Today's Appointments</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">
                    {appointments.filter(apt => 
                      new Date(apt.date).toDateString() === new Date().toDateString()
                    ).length}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Next Appointment</CardTitle>
                </CardHeader>
                <CardContent>
                  {appointments.length > 0 ? (
                    <>
                      <p className="text-xl font-medium">
                        {appointments
                          .filter(apt => new Date(apt.date) >= new Date())
                          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())[0]?.time || "No upcoming"}
                      </p>
                      <p className="text-sm text-gray-500">
                        {appointments
                          .filter(apt => new Date(apt.date) >= new Date())
                          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())[0]?.patient || "appointments"}
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-xl font-medium">No upcoming</p>
                      <p className="text-sm text-gray-500">appointments</p>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
            
            <div className="space-y-6">
                <Card className="w-full">
                <CardHeader>
                  <CardTitle>Upcoming Consultations</CardTitle>
                  <CardDescription>Scheduled for today</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {consultations.slice(0, 3).map((consultation) => (
                      <div key={consultation.id} className="flex items-center justify-between border-b pb-3">
                        <div className="flex items-center">
                          <Avatar className="h-9 w-9 mr-3">
                            <AvatarImage src="/placeholder.svg" alt={consultation.patient} />
                            <AvatarFallback>{consultation.patient.charAt(0)}</AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-medium">{consultation.patient}</p>
                            <div className="flex text-sm text-gray-500">
                              <span className="mr-3">{consultation.time}</span>
                              <span className="flex items-center">
                                {consultation.type === "Video" ? 
                                  <Video className="h-3 w-3 mr-1" /> : 
                                  <MessageSquare className="h-3 w-3 mr-1" />
                                }
                                {consultation.type}
                              </span>
                            </div>
                          </div>
                        </div>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleStartConsultation(consultation.id)}
                        >
                          Join
                        </Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" className="w-full" onClick={() => setActiveTab("consultations")}>
                    View All Consultations
                  </Button>
                </CardFooter>
              </Card>

            </div>
          </TabsContent>
          
          <TabsContent value="patients" className="mt-0">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold">Patient Management</h2>
              <div className="flex gap-2">
                <Input 
                  placeholder="Search patients..." 
                  className="w-64"
                />
                <Button>
                  Filter
                </Button>
              </div>
            </div>
            
            {isLoading ? (
              <div className="flex justify-center py-10">
                <Loader2 className="h-8 w-8 animate-spin text-health-600" />
              </div>
            ) : patients.length === 0 ? (
              <div className="text-center py-10 border rounded-lg">
                <p className="text-gray-500">No patients found</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {patients.map((patient) => (
                  <Card key={patient.id} className="overflow-hidden">
                    <CardContent className="p-0">
                      <div className="flex p-4">
                        <Avatar className="h-12 w-12 mr-4">
                          <AvatarImage src={patient.image} alt={patient.name} />
                          <AvatarFallback>{patient.name.charAt(0)}</AvatarFallback>
                        </Avatar>
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-bold text-lg">{patient.name}</h3>
                              <p className="text-gray-500 text-sm">
                                {patient.age} years • {patient.gender} • Last visit: {patient.lastVisit}
                              </p>
                            </div>
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium 
                              ${patient.status === "urgent" ? "bg-red-100 text-red-800" : 
                              patient.status === "scheduled" ? "bg-yellow-100 text-yellow-800" :
                              patient.status === "completed" ? "bg-green-100 text-green-800" :
                              "bg-blue-100 text-blue-800"}`}>
                              {patient.status}
                            </span>
                          </div>
                          
                          <div className="mt-2">
                            <p className="text-sm font-medium">Symptoms:</p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {patient.symptoms.map((symptom, index) => (
                                <span key={index} className="text-xs bg-gray-100 px-2 py-0.5 rounded">
                                  {symptom}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="border-t bg-gray-50 p-3 flex justify-between items-center">
                        <div className="text-sm">
                          <span className="font-medium">Next Appointment:</span> {patient.nextAppointment}
                        </div>
                        <div className="flex gap-2">
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => handleUploadPrescription(patient.id)}
                          >
                            <Upload className="h-4 w-4 mr-1" />
                            Prescription
                          </Button>
                          <Button 
                            size="sm"
                            onClick={() => handleStartConsultation(patient.id)}
                          >
                            <Video className="h-4 w-4 mr-1" />
                            Consult
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="consultations" className="mt-0">
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold">Upcoming Consultations</h2>
                <Button onClick={handleScheduleNew}>
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule New
                </Button>
              </div>
              
              {isLoading ? (
                <div className="flex justify-center py-10">
                  <Loader2 className="h-8 w-8 animate-spin text-health-600" />
                </div>
              ) : appointments.length === 0 ? (
                <div className="text-center py-10 border rounded-lg">
                  <p className="text-gray-500">No consultations scheduled</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {appointments
                    .filter(apt => apt.status === 'scheduled' || apt.status === 'confirmed')
                    .map((consultation) => (
                      <Card key={consultation.id}>
                        <CardHeader className="pb-2">
                          <div className="flex justify-between items-start">
                            <CardTitle className="text-lg">{consultation.patient}</CardTitle>
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium 
                              ${consultation.type === "Video" ? "bg-blue-100 text-blue-800" : "bg-purple-100 text-purple-800"}`}>
                              {consultation.type}
                            </span>
                          </div>
                          <CardDescription>{consultation.date}, {consultation.time}</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <p className="text-sm">{consultation.symptoms || "General consultation"}</p>
                          </div>
                        </CardContent>
                        <CardFooter className="gap-2">
                          <Button 
                            className="flex-1" 
                            variant="outline"
                            onClick={() => handleUploadPrescription(consultation.patientId)}
                          >
                            <Upload className="h-4 w-4 mr-1" />
                            Prescription
                          </Button>
                          <Button 
                            className="flex-1"
                            onClick={() => handleStartConsultation(consultation.id)}
                          >
                            {consultation.type === "Video" ? (
                              <><Video className="h-4 w-4 mr-1" /> Join</>
                            ) : (
                              <><MessageSquare className="h-4 w-4 mr-1" /> Call</>
                            )}
                          </Button>
                        </CardFooter>
                      </Card>
                    ))}
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="prescriptions" className="mt-0">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-bold mb-4">Prescriptions History</h2>
              <p className="text-gray-500">View and manage all uploaded prescriptions here.</p>
              
              <div className="mt-6">
                <div className="border rounded-lg overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="h-10 w-10 flex-shrink-0">
                              <img className="h-10 w-10 rounded-full" src="/placeholder.svg" alt="" />
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">Ananya Desai</div>
                              <div className="text-sm text-gray-500">29 years • Female</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3 days ago</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Medication</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            Dispensed
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button className="text-health-600 hover:text-health-900 mr-3">View</button>
                          <button className="text-health-600 hover:text-health-900">Edit</button>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="appointments" className="mt-0">
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold">Appointments Management</h2>
                <Button onClick={handleScheduleNew}>
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule New
                </Button>
              </div>
              
              {isLoading ? (
                <div className="flex justify-center py-10">
                  <Loader2 className="h-8 w-8 animate-spin text-health-600" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Today</CardTitle>
                      <CardDescription>{new Date().toLocaleDateString()}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {appointments
                        .filter(apt => new Date(apt.date).toDateString() === new Date().toDateString())
                        .map((appointment) => (
                          <div key={appointment.id} className="flex justify-between items-center p-3 border rounded-md bg-blue-50 hover:bg-blue-100 transition-colors">
                            <div>
                              <p className="font-medium">{appointment.patient}</p>
                              <div className="text-sm text-gray-600">{appointment.time}</div>
                            </div>
                            <Button 
                              size="sm" 
                              variant="ghost"
                              onClick={() => handleStartConsultation(appointment.id)}
                            >
                              View
                            </Button>
                          </div>
                        ))}
                      
                      {appointments.filter(apt => new Date(apt.date).toDateString() === new Date().toDateString()).length === 0 && (
                        <div className="text-center py-4 text-gray-500">
                          <p>No appointments today</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Tomorrow</CardTitle>
                      <CardDescription>{new Date(Date.now() + 86400000).toLocaleDateString()}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {appointments
                        .filter(apt => {
                          const tomorrow = new Date();
                          tomorrow.setDate(tomorrow.getDate() + 1);
                          return new Date(apt.date).toDateString() === tomorrow.toDateString();
                        })
                        .map((appointment) => (
                          <div key={appointment.id} className="flex justify-between items-center p-3 border rounded-md hover:bg-gray-50 transition-colors">
                            <div>
                              <p className="font-medium">{appointment.patient}</p>
                              <div className="text-sm text-gray-600">{appointment.time}</div>
                            </div>
                            <Button size="sm" variant="ghost">View</Button>
                          </div>
                        ))}
                        
                      {appointments.filter(apt => {
                        const tomorrow = new Date();
                        tomorrow.setDate(tomorrow.getDate() + 1);
                        return new Date(apt.date).toDateString() === tomorrow.toDateString();
                      }).length === 0 && (
                        <div className="text-center py-4 text-gray-500">
                          <p>No appointments tomorrow</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Upcoming Week</CardTitle>
                      <CardDescription>Next 7 days</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {appointments
                        .filter(apt => {
                          const appointmentDate = new Date(apt.date);
                          const today = new Date();
                          const nextWeek = new Date();
                          nextWeek.setDate(today.getDate() + 7);
                          return appointmentDate > today && appointmentDate <= nextWeek;
                        })
                        .map((appointment) => (
                          <div key={appointment.id} className="flex justify-between items-center p-3 border rounded-md hover:bg-gray-50 transition-colors">
                            <div>
                              <p className="font-medium">{appointment.patient}</p>
                              <div className="text-sm text-gray-600">{appointment.date}, {appointment.time}</div>
                            </div>
                            <Button size="sm" variant="ghost">View</Button>
                          </div>
                        ))}
                        
                      {appointments.filter(apt => {
                        const appointmentDate = new Date(apt.date);
                        const today = new Date();
                        const nextWeek = new Date();
                        nextWeek.setDate(today.getDate() + 7);
                        return appointmentDate > today && appointmentDate <= nextWeek;
                      }).length === 0 && (
                        <div className="text-center py-4 text-gray-500">
                          <p>No upcoming appointments</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="messages" className="mt-0">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-bold mb-4">Patient Messages</h2>
              <p className="text-gray-500">Secure messaging with patients would go here.</p>
            </div>
          </TabsContent>
          
          <TabsContent value="settings" className="mt-0">
            <DoctorProfile />
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Prescription Upload Dialog */}
      <Dialog open={showPrescriptionDialog} onOpenChange={setShowPrescriptionDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Upload Prescription</DialogTitle>
            <DialogDescription>
              Upload a handwritten or digital prescription for the patient.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <label htmlFor="prescription" className="text-sm font-medium">
                Prescription Image
              </label>
              <Input
                id="prescription"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
              />
              <p className="text-xs text-gray-500">Upload a clear image of the prescription (JPG, PNG)</p>
            </div>
            
            <div className="grid w-full gap-1.5">
              <label htmlFor="notes" className="text-sm font-medium">
                Additional Notes
              </label>
              <textarea 
                id="notes" 
                rows={3} 
                className="w-full rounded-md border border-gray-300 p-2 text-sm"
                placeholder="Add notes or instructions for the patient..."
                value={prescriptionNotes}
                onChange={(e) => setPrescriptionNotes(e.target.value)}
              ></textarea>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowPrescriptionDialog(false)}>Cancel</Button>
            <Button type="submit" onClick={handlePrescriptionSubmit}>Upload Prescription</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* New Appointment Dialog */}
      <Dialog open={showNewAppointmentDialog} onOpenChange={setShowNewAppointmentDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Schedule New Appointment</DialogTitle>
            <DialogDescription>
              Create a new appointment for a patient.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="grid w-full items-center gap-1.5">
              <label htmlFor="patientName" className="text-sm font-medium">
                Patient Name
              </label>
              <Input
                id="patientName"
                placeholder="Enter patient name"
                value={appointmentData.patientName}
                onChange={(e) => setAppointmentData({...appointmentData, patientName: e.target.value})}
              />
            </div>
            
            <div className="grid w-full items-center gap-1.5">
              <label htmlFor="appointmentDate" className="text-sm font-medium">
                Date
              </label>
              <Input
                id="appointmentDate"
                type="date"
                value={appointmentData.date}
                onChange={(e) => setAppointmentData({...appointmentData, date: e.target.value})}
              />
            </div>
            
            <div className="grid w-full items-center gap-1.5">
              <label htmlFor="appointmentTime" className="text-sm font-medium">
                Time
              </label>
              <Input
                id="appointmentTime"
                type="time"
                value={appointmentData.time}
                onChange={(e) => setAppointmentData({...appointmentData, time: e.target.value})}
              />
            </div>
            
            <div className="grid w-full items-center gap-1.5">
              <label htmlFor="appointmentType" className="text-sm font-medium">
                Appointment Type
              </label>
              <select
                id="appointmentType"
                className="w-full rounded-md border border-gray-300 p-2 text-sm"
                value={appointmentData.type}
                onChange={(e) => setAppointmentData({...appointmentData, type: e.target.value})}
              >
                <option value="Video">Video Consultation</option>
                <option value="Audio">Audio Call</option>
                <option value="In-Person">In-Person Visit</option>
              </select>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewAppointmentDialog(false)}>Cancel</Button>
            <Button type="submit" onClick={handleAppointmentSubmit}>Schedule Appointment</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default DoctorDashboard;