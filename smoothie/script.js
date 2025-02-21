// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add hover effect to CTA button
const ctaButton = document.querySelector('.cta-button');
if (ctaButton) {
    ctaButton.addEventListener('mouseenter', () => {
        ctaButton.style.transform = 'translateY(-2px)';
    });
    
    ctaButton.addEventListener('mouseleave', () => {
        ctaButton.style.transform = 'translateY(0)';
    });
}

// Add scroll effect to header
const header = document.querySelector('header');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll <= 0) {
        header.classList.remove('scroll-up');
        return;
    }
    
    if (currentScroll > lastScroll && !header.classList.contains('scroll-down')) {
        // Scroll down
        header.classList.remove('scroll-up');
        header.classList.add('scroll-down');
    } else if (currentScroll < lastScroll && header.classList.contains('scroll-down')) {
        // Scroll up
        header.classList.remove('scroll-down');
        header.classList.add('scroll-up');
    }
    
    lastScroll = currentScroll;
});

// OpenAI API configuration
const OPENAI_API_KEY = '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';

// Form state management
let formData = {
    age: '',
    weight: '',
    healthGoal: '',
    dietary: [],
    base: [],
    fruits: [],
    additions: []
};

// Multi-step form navigation
document.addEventListener('DOMContentLoaded', () => {
    const nextButtons = document.querySelectorAll('.next-step');
    const prevButtons = document.querySelectorAll('.prev-step');
    const submitButton = document.getElementById('get-smoothie-recommendation');

    // Next step buttons
    nextButtons.forEach(button => {
        button.addEventListener('click', () => {
            const currentStep = button.closest('.step-form');
            const nextStepId = button.dataset.next;
            
            // Validate current step
            if (validateStep(currentStep)) {
                // Save data from current step
                saveStepData(currentStep);
                
                // Update step indicators and show next step
                const currentStepNumber = parseInt(currentStep.id.replace('step', ''));
                const nextStepNumber = parseInt(nextStepId.replace('step', ''));
                
                document.querySelector(`.step-dot[data-step="${currentStepNumber}"]`).classList.remove('active');
                document.querySelector(`.step-dot[data-step="${nextStepNumber}"]`).classList.add('active');
                
                currentStep.style.display = 'none';
                document.getElementById(nextStepId).style.display = 'block';
            }
        });
    });

    // Previous step buttons
    prevButtons.forEach(button => {
        button.addEventListener('click', () => {
            const currentStep = button.closest('.step-form');
            const prevStepId = button.dataset.prev;
            
            // Update step indicators and show previous step
            const currentStepNumber = parseInt(currentStep.id.replace('step', ''));
            const prevStepNumber = parseInt(prevStepId.replace('step', ''));
            
            document.querySelector(`.step-dot[data-step="${currentStepNumber}"]`).classList.remove('active');
            document.querySelector(`.step-dot[data-step="${prevStepNumber}"]`).classList.add('active');
            
            currentStep.style.display = 'none';
            document.getElementById(prevStepId).style.display = 'block';
        });
    });

    // Submit form
    if (submitButton) {
        submitButton.addEventListener('click', async () => {
            const step5 = document.getElementById('step5');
            saveStepData(step5);
            
            // Show loading state
            const resultSection = document.getElementById('smoothie-recommendation-result');
            const loadingSpinner = resultSection.querySelector('.loading-spinner');
            const resultContent = resultSection.querySelector('.result-content');
            
            step5.style.display = 'none';
            resultSection.style.display = 'block';
            loadingSpinner.style.display = 'block';
            resultContent.innerHTML = '';

            try {
                const response = await fetch('http://localhost:3000/api/smoothie-recommendation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                if (!response.ok) {
                    throw new Error('API request failed');
                }

                const data = await response.json();
                const recommendation = data.choices[0].message.content;
                
                // Split recommendation into sections
                const sections = recommendation.split(/\d+\.\s+/).filter(Boolean);
                const sectionTitles = recommendation.match(/\d+\.\s+(.*?)(?=\n)/g).map(title => title.trim());
                
                // Create HTML for each section
                const formattedSections = sections.map((section, index) => {
                    const content = section
                        .replace(/^.*?\n/, '') // Remove title line
                        .replace(/(?:^|\n)-\s+(.*?)(?=(?:\n(?!-))|$)/g, '<li>$1</li>') // Convert bullet points to list items
                        .replace(/((?:<li>.*?<\/li>)+)/g, '<ul>$1</ul>') // Wrap list items in ul
                        .replace(/\n+/g, '<br>'); // Convert remaining newlines to breaks

                    return `
                        <div class="recommendation-section" style="display: none;" data-section="${index + 1}">
                            <h3>${sectionTitles[index]}</h3>
                            <div class="section-content">${content}</div>
                            <div class="section-navigation">
                                ${index > 0 ? `<button class="recommendation-button prev-section" data-prev="${index}">Previous</button>` : ''}
                                ${index < sections.length - 1 ? `<button class="recommendation-button next-section" data-next="${index + 2}">Next</button>` : ''}
                                ${index === sections.length - 1 ? `<button class="recommendation-button" onclick="resetForm()">Start Over</button>` : ''}
                            </div>
                        </div>
                    `;
                }).join('');

                resultContent.innerHTML = formattedSections;
                
                // Show first section
                document.querySelector('.recommendation-section[data-section="1"]').style.display = 'block';

                // Add navigation event listeners
                resultContent.addEventListener('click', (e) => {
                    if (e.target.classList.contains('next-section')) {
                        const nextSection = e.target.dataset.next;
                        document.querySelector(`.recommendation-section[data-section="${nextSection - 1}"]`).style.display = 'none';
                        document.querySelector(`.recommendation-section[data-section="${nextSection}"]`).style.display = 'block';
                    } else if (e.target.classList.contains('prev-section')) {
                        const prevSection = e.target.dataset.prev;
                        document.querySelector(`.recommendation-section[data-section="${prevSection + 1}"]`).style.display = 'none';
                        document.querySelector(`.recommendation-section[data-section="${prevSection}"]`).style.display = 'block';
                    }
                });
            } catch (error) {
                resultContent.innerHTML = 'Sorry, there was an error getting your recommendation. Please try again.';
                console.error('Error:', error);
            } finally {
                loadingSpinner.style.display = 'none';
            }
        });
    }
});

// Form validation
function validateStep(stepElement) {
    const inputs = stepElement.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value) {
            isValid = false;
            input.classList.add('invalid');
        } else {
            input.classList.remove('invalid');
        }
    });

    return isValid;
}

// Save form data
function saveStepData(stepElement) {
    if (stepElement.id === 'step1') {
        formData.age = document.getElementById('age').value;
        formData.weight = document.getElementById('weight').value;
        formData.healthGoal = document.getElementById('health-goal').value;
    } else if (stepElement.id === 'step2') {
        formData.dietary = Array.from(document.querySelectorAll('input[name="dietary"]:checked'))
            .map(checkbox => checkbox.value);
    } else if (stepElement.id === 'step3') {
        formData.base = Array.from(document.querySelectorAll('input[name="base"]:checked'))
            .map(checkbox => checkbox.value);
    } else if (stepElement.id === 'step4') {
        formData.fruits = Array.from(document.querySelectorAll('input[name="fruits"]:checked'))
            .map(checkbox => checkbox.value);
    } else if (stepElement.id === 'step5') {
        formData.additions = Array.from(document.querySelectorAll('input[name="additions"]:checked'))
            .map(checkbox => checkbox.value);
    }
}

// Reset form
function resetForm() {
    // Clear form data
    formData = {
        age: '',
        weight: '',
        healthGoal: '',
        dietary: [],
        base: [],
        fruits: [],
        additions: []
    };

    // Reset form elements
    document.getElementById('age').value = '';
    document.getElementById('weight').value = '';
    document.getElementById('health-goal').value = '';
    document.querySelectorAll('input[name="dietary"]').forEach(checkbox => checkbox.checked = false);
    document.querySelectorAll('input[name="base"]').forEach(checkbox => checkbox.checked = false);
    document.querySelectorAll('input[name="fruits"]').forEach(checkbox => checkbox.checked = false);
    document.querySelectorAll('input[name="additions"]').forEach(checkbox => checkbox.checked = false);

    // Reset step indicators and show first step
    document.querySelectorAll('.step-dot').forEach((dot, index) => {
        dot.classList.toggle('active', index === 0);
    });
    
    document.getElementById('step1').style.display = 'block';
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').style.display = 'none';
    document.getElementById('step4').style.display = 'none';
    document.getElementById('step5').style.display = 'none';
    document.getElementById('smoothie-recommendation-result').style.display = 'none';
}

// Smoothie cards animation
const smoothieCards = document.querySelectorAll('.smoothie-card');
smoothieCards.forEach((card, index) => {
    setTimeout(() => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
    }, 200 * index);
});
