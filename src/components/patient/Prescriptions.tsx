
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Pill, Calendar, Clock, Download, Plus, ArrowUpRight } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog";

interface Prescription {
  id: string;
  medication: string;
  dosage: string;
  frequency: string;
  duration: string;
  doctor: string;
  prescribedDate: string;
  instructions: string;
  refills: number;
}

const Prescriptions = () => {
  const { toast } = useToast();
  const [prescriptions, setPrescriptions] = useState<Prescription[]>([]);
  const [selectedPrescription, setSelectedPrescription] = useState<Prescription | null>(null);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [showRefillDialog, setShowRefillDialog] = useState(false);
  
  // Sample data
  useEffect(() => {
    const samplePrescriptions: Prescription[] = [
      {
        id: "1",
        medication: "Amoxicillin",
        dosage: "500mg",
        frequency: "3 times daily",
        duration: "7 days",
        doctor: "Dr. Sarah Johnson",
        prescribedDate: "2023-05-15",
        instructions: "Take with food. Complete the full course even if you feel better.",
        refills: 0
      },
      {
        id: "2",
        medication: "Lisinopril",
        dosage: "10mg",
        frequency: "Once daily",
        duration: "30 days",
        doctor: "Dr. Michael Chen",
        prescribedDate: "2023-05-01",
        instructions: "Take in the morning. Avoid grapefruit juice while on this medication.",
        refills: 3
      },
      {
        id: "3",
        medication: "Metformin",
        dosage: "1000mg",
        frequency: "Twice daily",
        duration: "90 days",
        doctor: "Dr. Lisa Patel",
        prescribedDate: "2023-04-20",
        instructions: "Take with meals to reduce stomach upset.",
        refills: 2
      }
    ];
    
    // Load from localStorage or use sample data
    const savedPrescriptions = localStorage.getItem('prescriptions');
    if (savedPrescriptions) {
      setPrescriptions(JSON.parse(savedPrescriptions));
    } else {
      setPrescriptions(samplePrescriptions);
      localStorage.setItem('prescriptions', JSON.stringify(samplePrescriptions));
    }
  }, []);
  
  const handleViewDetails = (prescription: Prescription) => {
    setSelectedPrescription(prescription);
    setShowDetailsDialog(true);
  };
  
  const handleRequestRefill = (prescription: Prescription) => {
    setSelectedPrescription(prescription);
    setShowRefillDialog(true);
  };
  
  const confirmRefill = () => {
    if (selectedPrescription) {
      toast({
        title: "Refill Requested",
        description: `Your refill request for ${selectedPrescription.medication} has been submitted.`,
      });
      setShowRefillDialog(false);
    }
  };
  
  const handleDownload = (prescription: Prescription) => {
    toast({
      title: "Prescription Downloaded",
      description: `${prescription.medication} prescription has been downloaded.`,
    });
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Your Prescriptions</h2>
          <p className="text-gray-500">Manage your medications and refills</p>
        </div>
      </div>
      
      {prescriptions.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {prescriptions.map((prescription) => (
            <Card key={prescription.id} className="shadow-sm hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="flex items-center gap-2">
                    <Pill className="h-5 w-5 text-health-600" />
                    {prescription.medication}
                  </CardTitle>
                  <span className="px-2 py-1 rounded-full text-xs bg-health-100 text-health-700">
                    {prescription.refills} refills left
                  </span>
                </div>
                <CardDescription>{prescription.dosage}, {prescription.frequency}</CardDescription>
              </CardHeader>
              <CardContent className="pb-3">
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Prescribed by:</span>
                    <span>{prescription.doctor}</span>
                  </div>
                  <div className="flex items-center text-sm">
                    <Calendar className="h-4 w-4 mr-1 text-gray-400" />
                    <span className="text-gray-500">Prescribed:</span>
                    <span className="ml-2">{prescription.prescribedDate}</span>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex flex-wrap gap-2 pt-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex-1"
                  onClick={() => handleViewDetails(prescription)}
                >
                  View Details
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex-1"
                  onClick={() => handleDownload(prescription)}
                >
                  <Download className="h-4 w-4 mr-1" />
                  Download
                </Button>
                {prescription.refills > 0 && (
                  <Button 
                    size="sm" 
                    className="w-full mt-2"
                    onClick={() => handleRequestRefill(prescription)}
                  >
                    Request Refill
                  </Button>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center p-8 border rounded-lg">
          <Pill className="h-12 w-12 mx-auto text-gray-300" />
          <p className="mt-2 text-gray-500">No prescriptions found</p>
        </div>
      )}
      
      {/* Prescription Details Dialog */}
      <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Prescription Details</DialogTitle>
          </DialogHeader>
          
          {selectedPrescription && (
            <div className="space-y-4 py-4">
              <div className="flex items-center gap-2">
                <Pill className="h-5 w-5 text-health-600" />
                <h3 className="text-lg font-semibold">{selectedPrescription.medication} {selectedPrescription.dosage}</h3>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Frequency</p>
                  <p className="font-medium">{selectedPrescription.frequency}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Duration</p>
                  <p className="font-medium">{selectedPrescription.duration}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Prescribed By</p>
                  <p className="font-medium">{selectedPrescription.doctor}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Prescribed Date</p>
                  <p className="font-medium">{selectedPrescription.prescribedDate}</p>
                </div>
                <div className="col-span-2">
                  <p className="text-sm text-gray-500">Instructions</p>
                  <p className="font-medium">{selectedPrescription.instructions}</p>
                </div>
                <div className="col-span-2">
                  <p className="text-sm text-gray-500">Refills Remaining</p>
                  <p className="font-medium">{selectedPrescription.refills}</p>
                </div>
              </div>
            </div>
          )}
          
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowDetailsDialog(false)}
            >
              Close
            </Button>
            <Button 
              onClick={() => handleDownload(selectedPrescription!)}
            >
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Refill Request Dialog */}
      <Dialog open={showRefillDialog} onOpenChange={setShowRefillDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Request Prescription Refill</DialogTitle>
            <DialogDescription>
              Confirm that you would like to request a refill for your medication.
            </DialogDescription>
          </DialogHeader>
          
          {selectedPrescription && (
            <div className="space-y-4 py-4">
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold">{selectedPrescription.medication} {selectedPrescription.dosage}</h3>
                <p className="text-sm text-gray-500">{selectedPrescription.frequency}</p>
                <p className="text-sm text-gray-500">Prescribed by {selectedPrescription.doctor}</p>
                <p className="text-sm font-medium mt-2">{selectedPrescription.refills} refills remaining</p>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm">Your refill will be processed and you will be notified when it's ready for pickup or delivery.</p>
              </div>
            </div>
          )}
          
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowRefillDialog(false)}
            >
              Cancel
            </Button>
            <Button 
              onClick={confirmRefill}
            >
              Confirm Refill Request
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Prescriptions;
