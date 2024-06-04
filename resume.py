import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class ResumeBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Builder")
        self.root.geometry("800x600")

        self.create_fields()
        self.create_buttons()
        self.create_templates()

    def create_fields(self):
        fields_frame = ttk.LabelFrame(self.root, text="Personal Information")
        fields_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.6)

        # Example fields, add more as needed
        labels = ["Name:", "Occupation:", "Experience:", "Degree:", "Summary:", "Skills:", "Interests:", "Projects:", "Social URLs:"]
        self.fields = {}

        for i, label_text in enumerate(labels):
            label = ttk.Label(fields_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(fields_frame)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            self.fields[label_text.strip(":")] = entry

    def create_buttons(self):
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.1)

        ttk.Button(buttons_frame, text="Create Resume", command=self.create_resume).pack(side="left", padx=10, pady=5)
        ttk.Button(buttons_frame, text="Clear Fields", command=self.clear_fields).pack(side="left", padx=10, pady=5)

    def create_templates(self):
        templates_frame = ttk.LabelFrame(self.root, text="Choose Template")
        templates_frame.place(relx=0.05, rely=0.825, relwidth=0.9, relheight=0.15)

        self.selected_template = tk.StringVar(value="Template 1")
        ttk.OptionMenu(templates_frame, self.selected_template, "Template 1", "Template 2", "Template 3").pack(padx=10, pady=5)

    def create_resume(self):
        template = self.selected_template.get()
        resume_content = self.get_resume_content()

        # Generate PDF
        filename = "resume.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        self.draw_template(template, c, resume_content)
        c.save()

        messagebox.showinfo("Resume Created", f"Resume has been created successfully as {filename}")

    def draw_template(self, template, c, resume_content):
        if template == "Template 1":
            y = 750
            for section, content in resume_content.items():
                c.drawString(100, y, f"{section}: {content}")
                y -= 20
        elif template == "Template 2":
            y = 750
            c.drawString(100, y, f"{resume_content['Name']}\t\t\t\t{resume_content['Occupation']}")
            y -= 30
            c.drawString(100, y, f"Summary:\n{resume_content['Summary']}")
            y -= 50
            c.drawString(100, y, f"Experience:\n{resume_content['Experience']}")
            y -= 20
            c.drawString(100, y, f"Skills:\n{resume_content['Skills']}")
            y -= 20
            c.drawString(100, y, f"Projects:\n{resume_content['Projects']}")
            y -= 20
            c.drawString(100, y, f"Interests:\n{resume_content['Interests']}")
            y -= 20
            c.drawString(100, y, f"Social URLs:\n{resume_content['Social URLs']}")
        elif template == "Template 3":
            y = 750
            c.drawString(100, y, f"{resume_content['Name']}\t\t\t\t{resume_content['Occupation']}")
            y -= 20
            c.drawString(100, y, f"Experience:\n{resume_content['Experience']}")
            y -= 30
            c.drawString(100, y, f"Summary:\n{resume_content['Summary']}")
            y -= 20
            c.drawString(100, y, f"Skills:\n{resume_content['Skills']}")
            y -= 20
            c.drawString(100, y, f"Interests:\n{resume_content['Interests']}")
            y -= 20
            c.drawString(100, y, f"Projects:\n{resume_content['Projects']}")
            y -= 20
            c.drawString(100, y, f"Social URLs:\n{resume_content['Social URLs']}")

    def get_resume_content(self):
        resume_content = {}
        for field, entry in self.fields.items():
            resume_content[field] = entry.get()
        return resume_content

    def clear_fields(self):
        for entry in self.fields.values():
            entry.delete(0, "end")

def main():
    root = tk.Tk()
    app = ResumeBuilder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
