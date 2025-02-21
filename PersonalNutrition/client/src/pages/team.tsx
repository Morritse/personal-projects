import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const teamMembers = [
  {
    name: "Alexandra Tran",
    role: "AI/ML Expert & Co-founder",
    education: "EECS, MIT",
    description: "Serial entrepreneur with success in launching automated food/beverage solutions. AI/ML expert with experience in international diplomacy and computer vision.",
    initials: "AT"
  },
  {
    name: "Evan Morritse",
    role: "Technical Lead & Co-founder",
    education: "Bioengineering, UC Berkeley",
    description: "Electromechanical design engineer for medical device startups. Lead AI/ML effort for J&J's surgical robotics program Ottava.",
    initials: "EM"
  },
  {
    name: "Betsy Tse",
    role: "Food Science Expert & Co-founder",
    education: "Food Science, Cornell",
    description: "Industry expert with over 30 years leading R&D for Odwalla and Coca-Cola health beverages. Lead external technology assessment for Coca-Cola and integrated within food/beverage startup space in bay area.",
    initials: "BT"
  }
];

export default function TeamPage() {
  return (
    <div className="container mx-auto py-12">
      <h1 className="text-4xl font-bold text-center mb-12">Our Team</h1>
      <p className="text-lg text-center text-muted-foreground mb-12">
        We are a group of engineers and scientists passionate about physical and mental health.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {teamMembers.map((member) => (
          <Card key={member.name} className="hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <Avatar className="w-24 h-24 mx-auto mb-4">
                <AvatarFallback className="text-xl bg-primary text-primary-foreground">
                  {member.initials}
                </AvatarFallback>
              </Avatar>
              <CardTitle className="mb-1">{member.name}</CardTitle>
              <div className="text-sm text-muted-foreground mb-2">{member.role}</div>
              <div className="text-sm font-medium text-primary">{member.education}</div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground text-sm leading-relaxed">
                {member.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
