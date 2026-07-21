import pandas as pd
import numpy as np
import os

def generate_salary_data(n_samples=5000):
    np.random.seed(42)
    
    # Features
    years_experience = np.random.uniform(0, 25, n_samples)
    
    education_levels = ['Bachelors', 'Masters', 'PhD']
    education = np.random.choice(education_levels, n_samples, p=[0.6, 0.3, 0.1])
    
    locations = ['San Francisco', 'New York', 'Austin', 'Remote', 'London']
    location = np.random.choice(locations, n_samples, p=[0.25, 0.25, 0.15, 0.25, 0.1])
    
    is_manager = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Base salary mapping by location
    base_salaries = {
        'San Francisco': 120000,
        'New York': 110000,
        'Remote': 90000,
        'Austin': 95000,
        'London': 80000
    }
    
    # Education multiplier
    edu_mult = {
        'Bachelors': 1.0,
        'Masters': 1.15,
        'PhD': 1.3
    }
    
    salaries = []
    for exp, edu, loc, mgr in zip(years_experience, education, location, is_manager):
        # Base salary for the location
        salary = base_salaries[loc]
        
        # Experience adds ~5k per year, with diminishing returns
        salary += (exp ** 0.8) * 5000
        
        # Education multiplier
        salary *= edu_mult[edu]
        
        # Manager bonus
        if mgr == 1:
            salary *= 1.2
            
        # Add some noise
        noise = np.random.normal(0, 10000)
        salary += noise
        
        salaries.append(max(40000, int(salary)))
        
    df = pd.DataFrame({
        'years_experience': np.round(years_experience, 1),
        'education_level': education,
        'location': location,
        'is_manager': is_manager,
        'salary': salaries
    })
    
    output_path = os.path.join(os.path.dirname(__file__), 'salary_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples and saved to {output_path}")

if __name__ == "__main__":
    generate_salary_data()
