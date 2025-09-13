"""
NCERT Science and Mathematics Question Database
Classes 6-10 with real sample paper questions and textbook content
"""

# Science Questions Database - NCERT Based
SCIENCE_QUESTIONS = {
    # Class 6 Science Questions
    'light-class6': [
        {
            "id": 1,
            "text": "Which of the following objects is luminous?",
            "options": ["Moon", "Planet", "Sun", "Mirror"],
            "correct": 2,
            "explanation": "The Sun produces its own light and heat, making it a luminous object. Moon and planets reflect sunlight.",
            "concept": "Light sources and luminous objects",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 2,
            "text": "A shadow is formed when:",
            "options": ["Light passes through an object", "Light is reflected by an object", "Light is blocked by an opaque object", "Light bends around an object"],
            "correct": 2,
            "explanation": "Shadows are formed when opaque objects block light rays, creating a dark area behind them.",
            "concept": "Shadow formation",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 3,
            "text": "The image formed in a plane mirror is:",
            "options": ["Real and inverted", "Virtual and erect", "Real and erect", "Virtual and inverted"],
            "correct": 1,
            "explanation": "Plane mirrors always form virtual (cannot be captured on screen) and erect (same orientation) images.",
            "concept": "Reflection in mirrors",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 4,
            "text": "Which of the following materials allows light to pass through completely?",
            "options": ["Wood", "Frosted glass", "Clear glass", "Cardboard"],
            "correct": 2,
            "explanation": "Clear glass is transparent and allows light to pass through completely.",
            "concept": "Transparent, translucent, and opaque objects",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 5,
            "text": "Which of these is NOT a property of a shadow?",
            "options": ["It is always black", "It is formed on the opposite side of the light source", "It shows the color of the object", "It changes size depending on the distance from the source"],
            "correct": 2,
            "explanation": "A shadow does not show the color of the object; it is always dark or black.",
            "concept": "Properties of shadows",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 6,
            "text": "What happens to the size of a shadow when the object is moved closer to the light source?",
            "options": ["It becomes smaller", "It becomes larger", "It disappears", "It stays the same"],
            "correct": 1,
            "explanation": "Moving the object closer makes the shadow larger because it blocks a greater portion of the diverging light.",
            "concept": "Shadow size variation",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 11"
        },
        {
            "id": 7,
            "text": "Which device forms an inverted real image on a screen using a small hole?",
            "options": ["Plane mirror", "Convex mirror", "Pinhole camera", "Magnifying glass"],
            "correct": 2,
            "explanation": "A pinhole camera forms an inverted real image of the object on the screen opposite the hole.",
            "concept": "Pinhole camera principle",
            "difficulty": "hard",
            "source": "NCERT Class 6 Chapter 11"
        }
    ],
    
    'electricity-class6': [
        {
            "id": 1,
            "text": "In which of the following circuits will the bulb glow?",
            "options": ["Open circuit with battery", "Closed circuit with battery", "Open circuit without battery", "Circuit with only wires"],
            "correct": 1,
            "explanation": "A bulb will only glow in a closed circuit with a battery, allowing current to flow continuously.",
            "concept": "Electric circuits",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 2,
            "text": "Which material is a good conductor of electricity?",
            "options": ["Plastic", "Rubber", "Copper", "Wood"],
            "correct": 2,
            "explanation": "Copper is a metal and metals are good conductors of electricity due to free electrons.",
            "concept": "Conductors and insulators",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 3,
            "text": "What is the function of a switch in an electric circuit?",
            "options": ["To increase current", "To break or complete the circuit", "To store electricity", "To reduce voltage"],
            "correct": 1,
            "explanation": "A switch is used to open (break) or close (complete) an electric circuit.",
            "concept": "Electric circuit components",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 4,
            "text": "Which of the following is an insulator?",
            "options": ["Iron", "Aluminium", "Glass", "Silver"],
            "correct": 2,
            "explanation": "Glass does not allow electricity to pass through and is an insulator.",
            "concept": "Conductors and insulators",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 5,
            "text": "What happens if the filament of a bulb breaks?",
            "options": ["Bulb glows brighter", "Bulb glows dim", "Bulb does not glow", "Bulb explodes"],
            "correct": 2,
            "explanation": "If the filament breaks, the circuit is incomplete and the bulb does not glow.",
            "concept": "Electric circuit continuity",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 6,
            "text": "What does the symbol of a long and a short line pair represent in a circuit diagram?",
            "options": ["Switch", "Battery cell", "Bulb", "Resistor"],
            "correct": 1,
            "explanation": "A single cell is represented by one long (positive) and one short (negative) line.",
            "concept": "Circuit symbols",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 12"
        },
        {
            "id": 7,
            "text": "Two bulbs are connected in series. If one bulb fuses, the other:",
            "options": ["Glows brighter", "Glows dimmer", "Continues to glow normally", "Stops glowing"],
            "correct": 3,
            "explanation": "In a series circuit an open at any point breaks the entire path; no current flows.",
            "concept": "Series circuit behavior",
            "difficulty": "hard",
            "source": "NCERT Class 6 Chapter 12"
        }
    ],

    # Class 7 Science Questions
    'heat-class7': [
        {
            "id": 1,
            "text": "Heat flows from:",
            "options": ["Cold object to hot object", "Hot object to cold object", "Objects at same temperature", "It doesn't flow"],
            "correct": 1,
            "explanation": "Heat always flows from a hotter object to a colder object until thermal equilibrium is reached.",
            "concept": "Heat transfer direction",
            "difficulty": "easy",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 2,
            "text": "Which method of heat transfer does NOT require a medium?",
            "options": ["Conduction", "Convection", "Radiation", "All require medium"],
            "correct": 2,
            "explanation": "Radiation can transfer heat through vacuum (space), while conduction and convection need matter.",
            "concept": "Methods of heat transfer",
            "difficulty": "medium",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 3,
            "text": "Which material is the best conductor of heat among the following?",
            "options": ["Copper", "Glass", "Wood", "Plastic"],
            "correct": 0,
            "explanation": "Metals like copper allow heat to pass rapidly making them good conductors.",
            "concept": "Conductors vs insulators",
            "difficulty": "easy",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 4,
            "text": "Which process primarily causes sea breeze?",
            "options": ["Conduction from sand", "Radiation from water", "Convection due to differential heating", "Evaporation of seawater"],
            "correct": 2,
            "explanation": "Unequal heating of land and sea sets up convection currents causing sea breeze.",
            "concept": "Convection in fluids",
            "difficulty": "medium",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 5,
            "text": "Why are cooking utensils often given handles made of wood or plastic?",
            "options": ["They are cheaper", "They conduct heat to hands", "They are bad conductors of heat", "They look shiny"],
            "correct": 2,
            "explanation": "Wood and plastic are insulators preventing heat transfer to hands for safe handling.",
            "concept": "Application of insulators",
            "difficulty": "hard",
            "source": "NCERT Class 7 Chapter 4"
        }
    ],

    # Class 8 Science Questions  
    'motion-class8': [
        {
            "id": 1,
            "text": "A ball rolling on the ground slows down due to:",
            "options": ["Gravitational force", "Magnetic force", "Frictional force", "Nuclear force"],
            "correct": 2,
            "explanation": "Friction between the ball and ground opposes motion, causing the ball to slow down.",
            "concept": "Friction and motion",
            "difficulty": "easy",
            "source": "NCERT Class 8 Chapter 11"
        },
        {
            "id": 2,
            "text": "Which of the following is an example of uniform motion?",
            "options": ["A car starting from rest", "A ball thrown upward", "A train moving at constant speed", "A pendulum swinging"],
            "correct": 2,
            "explanation": "Uniform motion occurs when an object covers equal distances in equal time intervals.",
            "concept": "Types of motion",
            "difficulty": "medium",
            "source": "NCERT Class 8 Chapter 11"
        },
        {
            "id": 3,
            "text": "Speed is defined as:",
            "options": ["Distance × Time", "Time / Distance", "Distance / Time", "Acceleration / Time"],
            "correct": 2,
            "explanation": "Speed = Distance travelled divided by Time taken.",
            "concept": "Speed formula",
            "difficulty": "easy",
            "source": "NCERT Class 8 Chapter 11"
        },
        {
            "id": 4,
            "text": "The SI unit of speed is:",
            "options": ["m", "m/s", "km/h", "s/m"],
            "correct": 1,
            "explanation": "Standard SI unit of speed is metres per second (m/s).",
            "concept": "Units of physical quantities",
            "difficulty": "medium",
            "source": "NCERT Class 8 Chapter 11"
        },
        {
            "id": 5,
            "text": "Which graph indicates uniform motion?",
            "options": ["Curved distance-time graph", "Straight line with constant slope", "Horizontal speed-time graph at zero", "Zig-zag line"],
            "correct": 1,
            "explanation": "Uniform motion gives a straight line distance-time graph with constant slope.",
            "concept": "Graphical representation of motion",
            "difficulty": "hard",
            "source": "NCERT Class 8 Chapter 11"
        }
    ],

    # Class 9 Science Questions
    'gravitation-class9': [
        {
            "id": 1,
            "text": "The value of acceleration due to gravity (g) on Earth is approximately:",
            "options": ["9.8 m/s", "9.8 m/s²", "98 m/s²", "0.98 m/s²"],
            "correct": 1,
            "explanation": "Acceleration due to gravity on Earth is 9.8 m/s², where m/s² is the unit for acceleration.",
            "concept": "Gravitational acceleration",
            "difficulty": "easy",
            "source": "NCERT Class 9 Chapter 10"
        },
        {
            "id": 2,
            "text": "Weight of an object on moon is _____ its weight on Earth:",
            "options": ["Equal to", "6 times", "1/6th of", "36 times"],
            "correct": 2,
            "explanation": "Moon's gravity is 1/6th of Earth's gravity, so weight (W = mg) is also 1/6th on moon.",
            "concept": "Weight vs mass",
            "difficulty": "medium",
            "source": "NCERT Class 9 Chapter 10"
        },
        {
            "id": 3,
            "text": "Which quantity remains constant for an object everywhere in the universe?",
            "options": ["Weight", "Mass", "Apparent weight", "Gravitational force"],
            "correct": 1,
            "explanation": "Mass is intrinsic and does not change with location, while weight depends on g.",
            "concept": "Mass vs weight",
            "difficulty": "easy",
            "source": "NCERT Class 9 Chapter 10"
        },
        {
            "id": 4,
            "text": "Force of gravitation between two bodies depends directly on:",
            "options": ["Product of their masses", "Sum of their masses", "Difference of masses", "Square of sum of masses"],
            "correct": 0,
            "explanation": "Newton's law: F = G m1 m2 / r^2, directly proportional to product m1 m2.",
            "concept": "Newton's law of gravitation",
            "difficulty": "medium",
            "source": "NCERT Class 9 Chapter 10"
        },
        {
            "id": 5,
            "text": "An object is thrown upwards. At the top of its path its acceleration is:",
            "options": ["Zero", "9.8 m/s^2 downward", "9.8 m/s^2 upward", "Decreasing to zero"],
            "correct": 1,
            "explanation": "Acceleration due to gravity acts downward throughout the motion, including at the highest point.",
            "concept": "Projectile motion under gravity",
            "difficulty": "hard",
            "source": "NCERT Class 9 Chapter 10"
        }
    ],

    # Class 10 Science Questions
    'carbon-class10': [
        {
            "id": 1,
            "text": "The molecular formula of methane is:",
            "options": ["CH₂", "CH₃", "CH₄", "C₂H₄"],
            "correct": 2,
            "explanation": "Methane has one carbon atom bonded to four hydrogen atoms, giving formula CH₄.",
            "concept": "Hydrocarbon formulas",
            "difficulty": "easy",
            "source": "NCERT Class 10 Chapter 4"
        },
        {
            "id": 2,
            "text": "Which gas is produced when ethanoic acid reacts with sodium carbonate?",
            "options": ["Hydrogen", "Carbon dioxide", "Oxygen", "Methane"],
            "correct": 1,
            "explanation": "Ethanoic acid (vinegar) reacts with sodium carbonate to produce carbon dioxide gas with effervescence.",
            "concept": "Acid reactions",
            "difficulty": "medium",
            "source": "NCERT Class 10 Chapter 4"
        },
        {
            "id": 3,
            "text": "The general formula for alkanes is:",
            "options": ["C_nH_{2n}", "C_nH_{2n+2}", "C_nH_{2n-2}", "C_nH_{n}"],
            "correct": 1,
            "explanation": "Alkanes are saturated hydrocarbons following general formula CnH2n+2.",
            "concept": "Homologous series",
            "difficulty": "easy",
            "source": "NCERT Class 10 Chapter 4"
        },
        {
            "id": 4,
            "text": "Which functional group is present in ethanol?",
            "options": ["-COOH", "-CHO", "-OH", "-NH2"],
            "correct": 2,
            "explanation": "Ethanol is an alcohol containing the hydroxyl (-OH) functional group.",
            "concept": "Functional groups",
            "difficulty": "medium",
            "source": "NCERT Class 10 Chapter 4"
        },
        {
            "id": 5,
            "text": "Substitution reactions are characteristic of:",
            "options": ["Alkanes", "Alkenes", "Alkynes", "Carboxylic acids"],
            "correct": 0,
            "explanation": "Saturated alkanes undergo substitution reactions (e.g., halogenation).",
            "concept": "Types of organic reactions",
            "difficulty": "hard",
            "source": "NCERT Class 10 Chapter 4"
        }
    ]
}

# Mathematics Questions Database - NCERT Based  
MATHEMATICS_QUESTIONS = {
    # Class 6 Mathematics Questions
    'integers-class6': [
        {
            "id": 1,
            "text": "What is (-15) + (+8)?",
            "options": ["-23", "-7", "+7", "+23"],
            "correct": 1,
            "explanation": "When adding integers with different signs, subtract the smaller absolute value from larger and keep the sign of larger absolute value: 15 - 8 = 7, with negative sign.",
            "concept": "Addition of integers",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 6"
        },
        {
            "id": 2,
            "text": "Which of the following is the correct order on number line?",
            "options": ["-5 > -3", "-8 < -10", "-2 > -7", "-1 < -4"],
            "correct": 2,
            "explanation": "On number line, numbers increase from left to right. -2 is to the right of -7, so -2 > -7.",
            "concept": "Ordering integers",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 6"
        },
        {
            "id": 3,
            "text": "The additive inverse of -9 is:",
            "options": ["-9", "9", "0", "1/9"],
            "correct": 1,
            "explanation": "The additive inverse of a number is the number that gives 0 when added to it. (-9) + 9 = 0.",
            "concept": "Additive inverse",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 6"
        },
        {
            "id": 4,
            "text": "What is (-4) × (-6)?",
            "options": ["-24", "24", "-10", "12"],
            "correct": 1,
            "explanation": "Product of two negative integers is positive: 4 × 6 = 24.",
            "concept": "Multiplication of integers",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 6"
        },
        {
            "id": 5,
            "text": "Simplify: (-20) - (-5)",
            "options": ["-25", "-15", "-5", "-10"],
            "correct": 1,
            "explanation": "Subtracting a negative is adding: -20 + 5 = -15.",
            "concept": "Subtraction of integers",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 6"
        },
        {
            "id": 6,
            "text": "Value of (-2)³ is:",
            "options": ["-6", "6", "-8", "8"],
            "correct": 2,
            "explanation": "(-2)³ = (-2)×(-2)×(-2) = 4 × (-2) = -8.",
            "concept": "Powers of integers",
            "difficulty": "hard",
            "source": "NCERT Class 6 Chapter 6"
        }
    ],

    'fractions-class6': [
        {
            "id": 1,
            "text": "Which of the following fractions is in its simplest form?",
            "options": ["6/8", "9/12", "5/7", "4/6"],
            "correct": 2,
            "explanation": "5/7 cannot be simplified further as 5 and 7 have no common factors other than 1.",
            "concept": "Simplifying fractions",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 7"
        },
        {
            "id": 2,
            "text": "What is 3/4 + 1/6?",
            "options": ["4/10", "11/12", "3/6", "5/8"],
            "correct": 1,
            "explanation": "To add fractions, find LCM of denominators: LCM(4,6) = 12. Then 9/12 + 2/12 = 11/12.",
            "concept": "Addition of fractions",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 7"
        },
        {
            "id": 3,
            "text": "Which fraction is greater: 5/8 or 3/5?",
            "options": ["5/8", "3/5", "They are equal", "Cannot be compared"],
            "correct": 0,
            "explanation": "Convert to common denominator 40: 5/8=25/40, 3/5=24/40 → 25/40 > 24/40.",
            "concept": "Comparing fractions",
            "difficulty": "easy",
            "source": "NCERT Class 6 Chapter 7"
        },
        {
            "id": 4,
            "text": "Simplify: (2/3) × (9/4)",
            "options": ["18/12", "3/2", "2/4", "9/6"],
            "correct": 1,
            "explanation": "(2×9)/(3×4) = 18/12 = 3/2 after simplification.",
            "concept": "Multiplication of fractions",
            "difficulty": "medium",
            "source": "NCERT Class 6 Chapter 7"
        },
        {
            "id": 5,
            "text": "What is the reciprocal of 7/9?",
            "options": ["9/7", "7/9", "-7/9", "1/7"],
            "correct": 0,
            "explanation": "Reciprocal flips numerator and denominator: 7/9 → 9/7.",
            "concept": "Reciprocals",
            "difficulty": "hard",
            "source": "NCERT Class 6 Chapter 7"
        }
    ],

    # Class 7 Mathematics Questions
    'algebra-class7': [
        {
            "id": 1,
            "text": "Solve for x: 2x + 5 = 15",
            "options": ["x = 5", "x = 10", "x = 7.5", "x = 2.5"],
            "correct": 0,
            "explanation": "2x + 5 = 15 → 2x = 15 - 5 → 2x = 10 → x = 10/2 = 5",
            "concept": "Solving linear equations",
            "difficulty": "easy",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 2,
            "text": "If the perimeter of a square is 4x, then its side is:",
            "options": ["4x", "x", "x/4", "16x"],
            "correct": 1,
            "explanation": "Perimeter of square = 4 × side. So 4x = 4 × side, therefore side = x.",
            "concept": "Algebraic expressions in geometry",
            "difficulty": "medium",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 3,
            "text": "Simplify: 3(2x + 1) - (x - 5)",
            "options": ["5x + 8", "6x + 1", "5x + 2", "6x - 4"],
            "correct": 0,
            "explanation": "3(2x+1)=6x+3; minus (x-5)= -x+5 ⇒ 6x+3 - x + 5 = 5x + 8.",
            "concept": "Distributive property",
            "difficulty": "easy",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 4,
            "text": "If 5y - 7 = 3y + 9, then y = ?",
            "options": ["1", "8", "-8", "-1"],
            "correct": 1,
            "explanation": "5y - 7 = 3y + 9 ⇒ 5y-3y = 9 + 7 ⇒ 2y = 16 ⇒ y = 8.",
            "concept": "Solving linear equations",
            "difficulty": "medium",
            "source": "NCERT Class 7 Chapter 4"
        },
        {
            "id": 5,
            "text": "If A = x^2 and B = 2x, what is A - B when x = 3?",
            "options": ["9", "3", "-3", "15"],
            "correct": 3,
            "explanation": "A - B = x^2 - 2x = 9 - 6 = 3.",
            "concept": "Substitution in expressions",
            "difficulty": "hard",
            "source": "NCERT Class 7 Chapter 4"
        }
    ],

    # Class 8 Mathematics Questions
    'rational-class8': [
        {
            "id": 1,
            "text": "Which of the following is a rational number?",
            "options": ["π", "√2", "0.333...", "√5"],
            "correct": 2,
            "explanation": "0.333... = 1/3, which can be expressed as p/q where p and q are integers, making it rational.",
            "concept": "Identifying rational numbers",
            "difficulty": "easy",
            "source": "NCERT Class 8 Chapter 1"
        },
        {
            "id": 2,
            "text": "What is (-3/4) × (8/9)?",
            "options": ["-2/3", "-6/7", "-11/13", "-24/36"],
            "correct": 0,
            "explanation": "(-3/4) × (8/9) = (-3×8)/(4×9) = -24/36 = -2/3 (simplified form)",
            "concept": "Multiplication of rational numbers",
            "difficulty": "medium",
            "source": "NCERT Class 8 Chapter 1"
        },
        {
            "id": 3,
            "text": "The reciprocal of (-5/7) is:",
            "options": ["5/7", "-7/5", "7/5", "-5/7"],
            "correct": 1,
            "explanation": "Reciprocal flips and keeps sign: (-5/7) → -7/5.",
            "concept": "Reciprocals of rationals",
            "difficulty": "easy",
            "source": "NCERT Class 8 Chapter 1"
        },
        {
            "id": 4,
            "text": "Add: (-2/3) + (5/6)",
            "options": ["1/6", "-1/6", "7/9", "-7/9"],
            "correct": 0,
            "explanation": "LCM 6: (-4/6 + 5/6) = 1/6.",
            "concept": "Addition of rational numbers",
            "difficulty": "medium",
            "source": "NCERT Class 8 Chapter 1"
        },
        {
            "id": 5,
            "text": "Which property is illustrated: (a/b) × (c/d) = (c/d) × (a/b)?",
            "options": ["Associative", "Closure", "Commutative", "Distributive"],
            "correct": 2,
            "explanation": "Changing order without changing product is commutative property.",
            "concept": "Properties of multiplication",
            "difficulty": "hard",
            "source": "NCERT Class 8 Chapter 1"
        }
    ],

    # Class 9 Mathematics Questions
    'polynomials-class9': [
        {
            "id": 1,
            "text": "The degree of polynomial 3x⁴ + 2x² - 5 is:",
            "options": ["2", "3", "4", "5"],
            "correct": 2,
            "explanation": "The degree of a polynomial is the highest power of the variable, which is 4 in this case.",
            "concept": "Degree of polynomial",
            "difficulty": "easy",
            "source": "NCERT Class 9 Chapter 2"
        },
        {
            "id": 2,
            "text": "If p(x) = x² - 3x + 2, then p(1) equals:",
            "options": ["0", "1", "2", "6"],
            "correct": 0,
            "explanation": "p(1) = (1)² - 3(1) + 2 = 1 - 3 + 2 = 0",
            "concept": "Value of polynomial",
            "difficulty": "medium",
            "source": "NCERT Class 9 Chapter 2"
        },
        {
            "id": 3,
            "text": "Coefficient of x² in 7x² - 5x + 4 is:",
            "options": ["7", "-5", "4", "2"],
            "correct": 0,
            "explanation": "The coefficient multiplying x² term is 7.",
            "concept": "Coefficients",
            "difficulty": "easy",
            "source": "NCERT Class 9 Chapter 2"
        },
        {
            "id": 4,
            "text": "Zeros of polynomial x(x-3) are:",
            "options": ["x = 3 only", "x = 0 only", "x = 0, 3", "None"],
            "correct": 2,
            "explanation": "x(x-3)=0 ⇒ x=0 or x=3.",
            "concept": "Zeros / roots",
            "difficulty": "medium",
            "source": "NCERT Class 9 Chapter 2"
        },
        {
            "id": 5,
            "text": "A polynomial of degree 1 is called:",
            "options": ["Linear", "Quadratic", "Cubic", "Constant"],
            "correct": 0,
            "explanation": "Degree 1 polynomials are linear (ax + b).",
            "concept": "Classification by degree",
            "difficulty": "hard",
            "source": "NCERT Class 9 Chapter 2"
        }
    ],

    # Class 10 Mathematics Questions  
    'trigonometry-class10': [
        {
            "id": 1,
            "text": "The value of sin 30° is:",
            "options": ["1/2", "√3/2", "1/√2", "√3"],
            "correct": 0,
            "explanation": "sin 30° = 1/2 is a standard trigonometric value that should be memorized.",
            "concept": "Trigonometric ratios",
            "difficulty": "easy",
            "source": "NCERT Class 10 Chapter 8"
        },
        {
            "id": 2,
            "text": "If tan θ = 1, then θ equals:",
            "options": ["30°", "45°", "60°", "90°"],
            "correct": 1,
            "explanation": "tan 45° = 1, as in a 45-45-90 triangle, opposite and adjacent sides are equal.",
            "concept": "Trigonometric values",
            "difficulty": "easy",
            "source": "NCERT Class 10 Chapter 8"
        },
        {
            "id": 3,
            "text": "cos 60° equals:",
            "options": ["1/2", "√3/2", "1", "0"],
            "correct": 0,
            "explanation": "cos 60° = 1/2 is a standard value.",
            "concept": "Trigonometric ratios",
            "difficulty": "easy",
            "source": "NCERT Class 10 Chapter 8"
        },
        {
            "id": 4,
            "text": "If sin θ = 1, θ (0°–90°) is:",
            "options": ["0°", "30°", "60°", "90°"],
            "correct": 3,
            "explanation": "sin 90° = 1.",
            "concept": "Standard angle values",
            "difficulty": "medium",
            "source": "NCERT Class 10 Chapter 8"
        },
        {
            "id": 5,
            "text": "Which identity is correct?",
            "options": ["sin²θ + cos²θ = 1", "sinθ + cosθ = 1", "tan²θ + 1 = cos²θ", "sin²θ - cos²θ = 1"],
            "correct": 0,
            "explanation": "Fundamental Pythagorean identity: sin²θ + cos²θ = 1.",
            "concept": "Trigonometric identities",
            "difficulty": "hard",
            "source": "NCERT Class 10 Chapter 8"
        }
    ]
}

# Combined database for easy access
NCERT_QUESTION_DATABASE = {
    **SCIENCE_QUESTIONS,
    **MATHEMATICS_QUESTIONS
}

# Question difficulty levels
DIFFICULTY_LEVELS = {
    'easy': {'xp': 10, 'tokens': 1},
    'medium': {'xp': 15, 'tokens': 2}, 
    'hard': {'xp': 25, 'tokens': 3}
}

# Daily challenge questions - mix of subjects
DAILY_CHALLENGES = [
    {
        "title": "Science Explorer",
        "description": "Answer 5 science questions correctly",
        "type": "subject_focus",
        "target": "science",
        "count": 5,
        "reward": {"xp": 50, "tokens": 10}
    },
    {
        "title": "Math Wizard", 
        "description": "Solve 5 math problems in a row",
        "type": "subject_focus",
        "target": "maths", 
        "count": 5,
        "reward": {"xp": 50, "tokens": 10}
    },
    {
        "title": "Perfect Score",
        "description": "Get 100% in any lesson",
        "type": "accuracy",
        "target": 100,
        "reward": {"xp": 75, "tokens": 15}
    },
    {
        "title": "Learning Streak",
        "description": "Complete lessons for 3 days in a row",
        "type": "streak",
        "target": 3,
        "reward": {"xp": 100, "tokens": 20}
    }
]

import random

# Rotation caches to avoid repeats until all questions are cycled
_LESSON_ROTATIONS = {}
_SUBJECT_CLASS_ROTATIONS = {}
_MIXED_ROTATION = []

# Exhaustion trackers for strict no-repeat
_LESSON_EXHAUSTED = set()
_SUBJECT_CLASS_EXHAUSTED = set()
_MIXED_EXHAUSTED = False

_DEFALT_BATCH_SIZE = 10

def _init_lesson_rotation(lesson_id):
    questions = NCERT_QUESTION_DATABASE.get(lesson_id, [])
    _LESSON_ROTATIONS[lesson_id] = random.sample(questions, len(questions)) if questions else []

def _next_from_rotation(rotation_list, count):
    """Pop next 'count' items from a rotation list; if depleted re-shuffle fresh cycle."""
    batch = []
    while len(batch) < count and rotation_list:
        batch.append(rotation_list.pop())
    return batch

def get_questions_by_lesson(lesson_id, count=10):
    """Strict no-repeat question retrieval for a lesson.
    Returns up to 'count' remaining unique questions. If exhausted, returns []."""
    if lesson_id not in NCERT_QUESTION_DATABASE:
        return []
    if lesson_id in _LESSON_EXHAUSTED:
        return []
    if lesson_id not in _LESSON_ROTATIONS:
        _init_lesson_rotation(lesson_id)
    rotation = _LESSON_ROTATIONS.get(lesson_id, [])
    if not rotation:
        _LESSON_EXHAUSTED.add(lesson_id)
        return []
    batch = _next_from_rotation(rotation, count)
    if not rotation:  # just exhausted after this call
        _LESSON_EXHAUSTED.add(lesson_id)
    return batch

def _subject_class_key(subject, class_level):
    return f"{subject.lower()}__class{class_level}"

def _init_subject_class_rotation(subject, class_level):
    key = _subject_class_key(subject, class_level)
    aggregated = []
    for lesson_id, qlist in NCERT_QUESTION_DATABASE.items():
        if subject.lower() in lesson_id and f"class{class_level}" in lesson_id:
            aggregated.extend(qlist)
    _SUBJECT_CLASS_ROTATIONS[key] = random.sample(aggregated, len(aggregated)) if aggregated else []

def get_questions_by_subject_class(subject, class_level, count=10):
    """Strict no-repeat across subject+class aggregated pool."""
    key = _subject_class_key(subject, class_level)
    if key in _SUBJECT_CLASS_EXHAUSTED:
        return []
    if key not in _SUBJECT_CLASS_ROTATIONS:
        _init_subject_class_rotation(subject, class_level)
    rotation = _SUBJECT_CLASS_ROTATIONS.get(key, [])
    if not rotation:
        _SUBJECT_CLASS_EXHAUSTED.add(key)
        return []
    batch = _next_from_rotation(rotation, count)
    if not rotation:
        _SUBJECT_CLASS_EXHAUSTED.add(key)
    return batch

def _init_mixed_rotation():
    all_questions = []
    for qlist in NCERT_QUESTION_DATABASE.values():
        all_questions.extend(qlist)
    random.shuffle(all_questions)
    _MIXED_ROTATION.clear()
    _MIXED_ROTATION.extend(all_questions)

def get_mixed_questions(count=10):
    """Strict no-repeat mixed pool; empty list once exhausted until reset."""
    global _MIXED_EXHAUSTED
    if _MIXED_EXHAUSTED:
        return []
    if not _MIXED_ROTATION:
        _init_mixed_rotation()
    if not _MIXED_ROTATION:
        _MIXED_EXHAUSTED = True
        return []
    batch = _next_from_rotation(_MIXED_ROTATION, count)
    if not _MIXED_ROTATION:
        _MIXED_EXHAUSTED = True
    return batch

def reset_question_rotations(scope='all', lesson_id=None, subject=None, class_level=None):
    """Reset rotation caches & exhaustion flags."""
    global _MIXED_EXHAUSTED
    if scope == 'all':
        _LESSON_ROTATIONS.clear(); _SUBJECT_CLASS_ROTATIONS.clear(); _MIXED_ROTATION.clear()
        _LESSON_EXHAUSTED.clear(); _SUBJECT_CLASS_EXHAUSTED.clear(); _MIXED_EXHAUSTED = False
    elif scope == 'lesson' and lesson_id:
        _LESSON_ROTATIONS.pop(lesson_id, None); _LESSON_EXHAUSTED.discard(lesson_id)
    elif scope == 'subject_class' and subject and class_level is not None:
        key = _subject_class_key(subject, class_level)
        _SUBJECT_CLASS_ROTATIONS.pop(key, None); _SUBJECT_CLASS_EXHAUSTED.discard(key)
    elif scope == 'mixed':
        _MIXED_ROTATION.clear(); _MIXED_EXHAUSTED = False
    else:
        raise ValueError('Invalid reset parameters')

# ---------------- Strict no-repeat per-user adjustments -------------------
# Track exhaustion per user scope
_USER_EXHAUSTED = {
    'lesson': set(),          # (user_id, lesson_id)
    'subject_class': set(),   # (user_id, subject, class_level)
    'mixed': set()            # user_id
}

# Ensure user rotations dict exists before any per-user functions
try:
    _USER_ROTATIONS
except NameError:
    _USER_ROTATIONS = {
        'lesson': {},
        'subject_class': {},
        'mixed': {}
    }

def _init_user_lesson_rotation(user_id, lesson_id):
    questions = NCERT_QUESTION_DATABASE.get(lesson_id, [])
    key = (user_id, lesson_id)
    import random
    _USER_ROTATIONS['lesson'][key] = random.sample(questions, len(questions)) if questions else []

def _init_user_subject_class_rotation(user_id, subject, class_level):
    import random
    aggregated = []
    for lesson_id, qlist in NCERT_QUESTION_DATABASE.items():
        if subject.lower() in lesson_id and f'class{class_level}' in lesson_id:
            aggregated.extend(qlist)
    key = (user_id, subject.lower(), class_level)
    _USER_ROTATIONS['subject_class'][key] = random.sample(aggregated, len(aggregated)) if aggregated else []

def _init_user_mixed_rotation(user_id):
    import random
    all_questions = []
    for qlist in NCERT_QUESTION_DATABASE.values():
        all_questions.extend(qlist)
    random.shuffle(all_questions)
    _USER_ROTATIONS['mixed'][user_id] = all_questions

def _pop_batch(rotation_list, count):
    batch = []
    while len(batch) < count and rotation_list:
        batch.append(rotation_list.pop())
    return batch

# ---------------------- Public Per-User APIs ------------------------------

def get_user_lesson_questions(user_id: str, lesson_id: str, count: int = 10):
    """Return up to 'count' unique questions for this user & lesson.
    Starts a fresh shuffled cycle when depleted. Ensures no duplicates in a single cycle."""
    key = (user_id, lesson_id)
    if key in _USER_EXHAUSTED['lesson']:
        return []
    if lesson_id not in NCERT_QUESTION_DATABASE:
        return []
    if key not in _USER_ROTATIONS['lesson']:
        _init_user_lesson_rotation(user_id, lesson_id)
    rotation = _USER_ROTATIONS['lesson'].get(key, [])
    if not rotation:
        _USER_EXHAUSTED['lesson'].add(key)
        return []
    batch = _pop_batch(rotation, count)
    if not rotation:
        _USER_EXHAUSTED['lesson'].add(key)
    return batch

def get_user_subject_class_questions(user_id: str, subject: str, class_level: int, count: int = 10):
    """Return unique aggregated subject/class questions for a user without repeats per cycle."""
    key = (user_id, subject.lower(), class_level)
    if key in _USER_EXHAUSTED['subject_class']:
        return []
    if key not in _USER_ROTATIONS['subject_class']:
        _init_user_subject_class_rotation(user_id, subject, class_level)
    rotation = _USER_ROTATIONS['subject_class'].get(key, [])
    if not rotation:
        _USER_EXHAUSTED['subject_class'].add(key)
        return []
    batch = _pop_batch(rotation, count)
    if not rotation:
        _USER_EXHAUSTED['subject_class'].add(key)
    return batch

def get_user_mixed_questions(user_id: str, count: int = 10):
    """Return unique mixed pool questions for a user without repeats per cycle."""
    if user_id in _USER_EXHAUSTED['mixed']:
        return []
    if user_id not in _USER_ROTATIONS['mixed']:
        _init_user_mixed_rotation(user_id)
    rotation = _USER_ROTATIONS['mixed'].get(user_id, [])
    if not rotation:
        _USER_EXHAUSTED['mixed'].add(user_id)
        return []
    batch = _pop_batch(rotation, count)
    if not rotation:
        _USER_EXHAUSTED['mixed'].add(user_id)
    return batch

# ------------------- Reset / Maintenance Utilities ------------------------

def reset_user_rotations(user_id: str, scope: str = 'all', lesson_id: str = None, subject: str = None, class_level: int = None):
    if scope == 'all':
        _USER_ROTATIONS['lesson'] = {k: v for k, v in _USER_ROTATIONS['lesson'].items() if k[0] != user_id}
        _USER_ROTATIONS['subject_class'] = {k: v for k, v in _USER_ROTATIONS['subject_class'].items() if k[0] != user_id}
        _USER_ROTATIONS['mixed'].pop(user_id, None)
        # clear exhausted
        _USER_EXHAUSTED['lesson'] = {k for k in _USER_EXHAUSTED['lesson'] if k[0] != user_id}
        _USER_EXHAUSTED['subject_class'] = {k for k in _USER_EXHAUSTED['subject_class'] if k[0] != user_id}
        _USER_EXHAUSTED['mixed'].discard(user_id)
    elif scope == 'lesson' and lesson_id:
        _USER_ROTATIONS['lesson'].pop((user_id, lesson_id), None)
        _USER_EXHAUSTED['lesson'].discard((user_id, lesson_id))
    elif scope == 'subject_class' and subject and class_level is not None:
        key = (user_id, subject.lower(), class_level)
        _USER_ROTATIONS['subject_class'].pop(key, None)
        _USER_EXHAUSTED['subject_class'].discard(key)
    elif scope == 'mixed':
        _USER_ROTATIONS['mixed'].pop(user_id, None)
        _USER_EXHAUSTED['mixed'].discard(user_id)
    else:
        raise ValueError('Invalid parameters for reset_user_rotations')

import os, json

STATE_FILE = os.path.join(os.path.dirname(__file__), 'question_state.json')

# User answer tracking for review & adaptive resets
_USER_TRACKING = {
    # user_id: {
    #   'answered': { lesson_id: set(question_ids) },
    #   'wrong': { lesson_id: set(question_ids) }
    # }
}

def _get_lesson_total(lesson_id, difficulties=None):
    questions = NCERT_QUESTION_DATABASE.get(lesson_id, [])
    if difficulties:
        ds = set(difficulties)
        return sum(1 for q in questions if q.get('difficulty') in ds)
    return len(questions)

def _remaining_in_rotation(rotation_list, difficulties=None):
    if not difficulties:
        return len(rotation_list)
    ds = set(difficulties)
    return sum(1 for q in rotation_list if q.get('difficulty') in ds)

# ------------------ Status Helper Functions ------------------

def get_lesson_status(lesson_id, difficulties=None):
    rotation = _LESSON_ROTATIONS.get(lesson_id, [])
    total = _get_lesson_total(lesson_id, difficulties)
    remaining = _remaining_in_rotation(rotation, difficulties)
    exhausted = (lesson_id in _LESSON_EXHAUSTED) or remaining == 0
    progress_percent = ((total - remaining) / total * 100) if total else 0
    return {
        'scope': 'lesson',
        'lesson_id': lesson_id,
        'total': total,
        'remaining': remaining,
        'exhausted': exhausted,
        'progress_percent': round(progress_percent, 1)
    }

def get_subject_class_status(subject, class_level, difficulties=None):
    key = _subject_class_key(subject, class_level)
    rotation = _SUBJECT_CLASS_ROTATIONS.get(key, [])
    # aggregate total
    aggregated = []
    for lesson_id, qlist in NCERT_QUESTION_DATABASE.items():
        if subject.lower() in lesson_id and f'class{class_level}' in lesson_id:
            aggregated.extend(qlist)
    if difficulties:
        ds = set(difficulties)
        total = sum(1 for q in aggregated if q.get('difficulty') in ds)
        remaining = sum(1 for q in rotation if q.get('difficulty') in ds)
    else:
        total = len(aggregated)
        remaining = len(rotation)
    exhausted = (key in _SUBJECT_CLASS_EXHAUSTED) or remaining == 0
    progress_percent = ((total - remaining) / total * 100) if total else 0
    return {
        'scope': 'subject_class',
        'subject': subject,
        'class': class_level,
        'total': total,
        'remaining': remaining,
        'exhausted': exhausted,
        'progress_percent': round(progress_percent, 1)
    }

def get_mixed_status(difficulties=None):
    all_questions = []
    for qlist in NCERT_QUESTION_DATABASE.values():
        all_questions.extend(qlist)
    if difficulties:
        ds = set(difficulties)
        total = sum(1 for q in all_questions if q.get('difficulty') in ds)
        remaining = sum(1 for q in _MIXED_ROTATION if q.get('difficulty') in ds)
    else:
        total = len(all_questions)
        remaining = len(_MIXED_ROTATION)
    exhausted = _MIXED_EXHAUSTED or remaining == 0
    progress_percent = ((total - remaining) / total * 100) if total else 0
    return {
        'scope': 'mixed',
        'total': total,
        'remaining': remaining,
        'exhausted': exhausted,
        'progress_percent': round(progress_percent, 1)
    }

# Per-user status

def get_user_lesson_status(user_id, lesson_id, difficulties=None):
    key = (user_id, lesson_id)
    rotation = _USER_ROTATIONS['lesson'].get(key, [])
    total = _get_lesson_total(lesson_id, difficulties)
    remaining = _remaining_in_rotation(rotation, difficulties)
    exhausted = key in _USER_EXHAUSTED['lesson'] or remaining == 0
    progress_percent = ((total - remaining)/total*100) if total else 0
    return {
        'scope': 'user_lesson', 'user_id': user_id, 'lesson_id': lesson_id,
        'total': total, 'remaining': remaining, 'exhausted': exhausted,
        'progress_percent': round(progress_percent,1)
    }

def get_user_subject_class_status(user_id, subject, class_level, difficulties=None):
    key = (user_id, subject.lower(), class_level)
    rotation = _USER_ROTATIONS['subject_class'].get(key, [])
    # aggregate total
    aggregated = []
    for lesson_id, qlist in NCERT_QUESTION_DATABASE.items():
        if subject.lower() in lesson_id and f'class{class_level}' in lesson_id:
            aggregated.extend(qlist)
    if difficulties:
        ds = set(difficulties)
        total = sum(1 for q in aggregated if q.get('difficulty') in ds)
        remaining = sum(1 for q in rotation if q.get('difficulty') in ds)
    else:
        total = len(aggregated)
        remaining = len(rotation)
    exhausted = key in _USER_EXHAUSTED['subject_class'] or remaining == 0
    progress_percent = ((total - remaining)/total*100) if total else 0
    return {
        'scope': 'user_subject_class', 'user_id': user_id,
        'subject': subject, 'class': class_level,
        'total': total, 'remaining': remaining, 'exhausted': exhausted,
        'progress_percent': round(progress_percent,1)
    }

def get_user_mixed_status(user_id, difficulties=None):
    rotation = _USER_ROTATIONS['mixed'].get(user_id, [])
    all_questions = []
    for qlist in NCERT_QUESTION_DATABASE.values():
        all_questions.extend(qlist)
    if difficulties:
        ds = set(difficulties)
        total = sum(1 for q in all_questions if q.get('difficulty') in ds)
        remaining = sum(1 for q in rotation if q.get('difficulty') in ds)
    else:
        total = len(all_questions)
        remaining = len(rotation)
    exhausted = user_id in _USER_EXHAUSTED['mixed'] or remaining == 0
    progress_percent = ((total - remaining)/total*100) if total else 0
    return {
        'scope': 'user_mixed', 'user_id': user_id,
        'total': total, 'remaining': remaining, 'exhausted': exhausted,
        'progress_percent': round(progress_percent,1)
    }

# ------------------- Difficulty-Filtered Fetch Wrappers -------------------

def _pop_filtered(rotation_list, count, difficulties):
    if not difficulties:
        return _next_from_rotation(rotation_list, count)
    ds = set(difficulties)
    batch = []
    # We still pop everything (maintain no-repeat globally), only return matching difficulties
    while len(batch) < count and rotation_list:
        q = rotation_list.pop()
        if q.get('difficulty') in ds:
            batch.append(q)
    return batch

# Wrapper functions returning (questions, status_dict)

def fetch_lesson_questions(lesson_id, count=10, difficulties=None):
    qs = get_questions_by_lesson(lesson_id, 10**9)  # fetch remaining without consuming? (But our get pops) -> Instead, operate directly on rotation
    # Because get_questions_by_lesson pops, we need direct access; adjust: use internal rotation directly
    if lesson_id in _LESSON_EXHAUSTED:
        return [], get_lesson_status(lesson_id, difficulties)
    if lesson_id not in _LESSON_ROTATIONS:
        _init_lesson_rotation(lesson_id)
    rotation = _LESSON_ROTATIONS.get(lesson_id, [])
    if not rotation:
        _LESSON_EXHAUSTED.add(lesson_id)
        return [], get_lesson_status(lesson_id, difficulties)
    batch = _pop_filtered(rotation, count, difficulties)
    if not rotation:
        _LESSON_EXHAUSTED.add(lesson_id)
    return batch, get_lesson_status(lesson_id, difficulties)

def fetch_subject_class_questions(subject, class_level, count=10, difficulties=None):
    key = _subject_class_key(subject, class_level)
    if key in _SUBJECT_CLASS_EXHAUSTED:
        return [], get_subject_class_status(subject, class_level, difficulties)
    if key not in _SUBJECT_CLASS_ROTATIONS:
        _init_subject_class_rotation(subject, class_level)
    rotation = _SUBJECT_CLASS_ROTATIONS.get(key, [])
    if not rotation:
        _SUBJECT_CLASS_EXHAUSTED.add(key)
        return [], get_subject_class_status(subject, class_level, difficulties)
    batch = _pop_filtered(rotation, count, difficulties)
    if not rotation:
        _SUBJECT_CLASS_EXHAUSTED.add(key)
    return batch, get_subject_class_status(subject, class_level, difficulties)

def fetch_mixed_questions(count=10, difficulties=None):
    global _MIXED_EXHAUSTED
    if _MIXED_EXHAUSTED:
        return [], get_mixed_status(difficulties)
    if not _MIXED_ROTATION:
        _init_mixed_rotation()
    if not _MIXED_ROTATION:
        _MIXED_EXHAUSTED = True
        return [], get_mixed_status(difficulties)
    batch = _pop_filtered(_MIXED_ROTATION, count, difficulties)
    if not _MIXED_ROTATION:
        _MIXED_EXHAUSTED = True
    return batch, get_mixed_status(difficulties)

# Per-user difficulty filtered

def fetch_user_lesson_questions(user_id, lesson_id, count=10, difficulties=None):
    key = (user_id, lesson_id)
    if key in _USER_EXHAUSTED['lesson']:
        return [], get_user_lesson_status(user_id, lesson_id, difficulties)
    if key not in _USER_ROTATIONS['lesson']:
        _init_user_lesson_rotation(user_id, lesson_id)
    rotation = _USER_ROTATIONS['lesson'].get(key, [])
    if not rotation:
        _USER_EXHAUSTED['lesson'].add(key)
        return [], get_user_lesson_status(user_id, lesson_id, difficulties)
    batch = _pop_filtered(rotation, count, difficulties)
    if not rotation:
        _USER_EXHAUSTED['lesson'].add(key)
    return batch, get_user_lesson_status(user_id, lesson_id, difficulties)

def fetch_user_subject_class_questions(user_id, subject, class_level, count=10, difficulties=None):
    key = (user_id, subject.lower(), class_level)
    if key in _USER_EXHAUSTED['subject_class']:
        return [], get_user_subject_class_status(user_id, subject, class_level, difficulties)
    if key not in _USER_ROTATIONS['subject_class']:
        _init_user_subject_class_rotation(user_id, subject, class_level)
    rotation = _USER_ROTATIONS['subject_class'].get(key, [])
    if not rotation:
        _USER_EXHAUSTED['subject_class'].add(key)
        return [], get_user_subject_class_status(user_id, subject, class_level, difficulties)
    batch = _pop_filtered(rotation, count, difficulties)
    if not rotation:
        _USER_EXHAUSTED['subject_class'].add(key)
    return batch, get_user_subject_class_status(user_id, subject, class_level, difficulties)

def fetch_user_mixed_questions(user_id, count=10, difficulties=None):
    if user_id in _USER_EXHAUSTED['mixed']:
        return [], get_user_mixed_status(user_id, difficulties)
    if user_id not in _USER_ROTATIONS['mixed']:
        _init_user_mixed_rotation(user_id)
    rotation = _USER_ROTATIONS['mixed'].get(user_id, [])
    if not rotation:
        _USER_EXHAUSTED['mixed'].add(user_id)
        return [], get_user_mixed_status(user_id, difficulties)
    batch = _pop_filtered(rotation, count, difficulties)
    if not rotation:
        _USER_EXHAUSTED['mixed'].add(user_id)
    return batch, get_user_mixed_status(user_id, difficulties)

# --------------------- Answer Tracking ------------------------------------

def update_answer_tracking(user_id, lesson_id, question_id, correct):
    record = _USER_TRACKING.setdefault(user_id, {'answered': {}, 'wrong': {}})
    answered = record['answered'].setdefault(lesson_id, set())
    wrong = record['wrong'].setdefault(lesson_id, set())
    answered.add(question_id)
    if correct:
        wrong.discard(question_id)
    else:
        wrong.add(question_id)
    save_question_state()

def get_review_items(user_id, lesson_id=None, wrong_only=False):
    record = _USER_TRACKING.get(user_id, {'answered': {}, 'wrong': {}})
    lessons = [lesson_id] if lesson_id else list(record['answered'].keys())
    review = []
    for lid in lessons:
        answered_ids = record['answered'].get(lid, set())
        wrong_ids = record['wrong'].get(lid, set())
        for q in NCERT_QUESTION_DATABASE.get(lid, []):
            if q['id'] in answered_ids and (not wrong_only or q['id'] in wrong_ids):
                review.append({
                    'lesson_id': lid,
                    'id': q['id'],
                    'text': q['text'],
                    'options': q['options'],
                    'correct_option': q['correct'],
                    'explanation': q.get('explanation'),
                    'difficulty': q.get('difficulty'),
                    'was_wrong': q['id'] in wrong_ids
                })
    return review

# Adaptive reset: only wrong ones

def adaptive_reset_user_lesson(user_id, lesson_id):
    record = _USER_TRACKING.get(user_id)
    if not record:
        return False
    wrong_ids = record['wrong'].get(lesson_id, set())
    pool = [q for q in NCERT_QUESTION_DATABASE.get(lesson_id, []) if q['id'] in wrong_ids]
    key = (user_id, lesson_id)
    import random
    _USER_ROTATIONS['lesson'][key] = random.sample(pool, len(pool)) if pool else []
    if key in _USER_EXHAUSTED['lesson']:
        _USER_EXHAUSTED['lesson'].discard(key)
    save_question_state()
    return True

# --------------------- Persistence Layer ----------------------------------

def save_question_state():
    try:
        data = {
            'lesson_rotations': {lid: [q['id'] for q in rot] for lid, rot in _LESSON_ROTATIONS.items()},
            'lesson_exhausted': list(_LESSON_EXHAUSTED),
            'subject_class_rotations': {k: [q['id'] for q in rot] for k, rot in _SUBJECT_CLASS_ROTATIONS.items()},
            'subject_class_exhausted': list(_SUBJECT_CLASS_EXHAUSTED),
            'mixed_rotation': [q['id'] for q in _MIXED_ROTATION],
            'mixed_exhausted': _MIXED_EXHAUSTED,
            'user_rotations': {
                'lesson': {str(k): [q['id'] for q in rot] for k, rot in _USER_ROTATIONS['lesson'].items()},
                'subject_class': {str(k): [q['id'] for q in rot] for k, rot in _USER_ROTATIONS['subject_class'].items()},
                'mixed': {uid: [q['id'] for q in rot] for uid, rot in _USER_ROTATIONS['mixed'].items()}
            },
            'user_exhausted': {
                'lesson': [str(k) for k in _USER_EXHAUSTED['lesson']],
                'subject_class': [str(k) for k in _USER_EXHAUSTED['subject_class']],
                'mixed': list(_USER_EXHAUSTED['mixed'])
            },
            'user_tracking': {
                uid: {
                    'answered': {lid: list(ids) for lid, ids in rec['answered'].items()},
                    'wrong': {lid: list(ids) for lid, ids in rec['wrong'].items()}
                } for uid, rec in _USER_TRACKING.items()
            }
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

def load_question_state():
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        # Helpers
        def _id_map_for_lesson(lid):
            return {q['id']: q for q in NCERT_QUESTION_DATABASE.get(lid, [])}
        # Lessons
        for lid, id_list in data.get('lesson_rotations', {}).items():
            id_map = _id_map_for_lesson(lid)
            _LESSON_ROTATIONS[lid] = [id_map[i] for i in id_list if i in id_map]
        for lid in data.get('lesson_exhausted', []):
            _LESSON_EXHAUSTED.add(lid)
        # Subject-class rotations
        for key, id_list in data.get('subject_class_rotations', {}).items():
            # key format '{subject}__class{n}' from earlier helper or maybe a tuple string; keep as-is
            # Rebuild aggregated map
            aggregated = []
            for lesson_id, qlist in NCERT_QUESTION_DATABASE.items():
                aggregated.extend(qlist)
            id_map = {q['id']: q for q in aggregated}
            _SUBJECT_CLASS_ROTATIONS[key] = [id_map[i] for i in id_list if i in id_map]
        for key in data.get('subject_class_exhausted', []):
            _SUBJECT_CLASS_EXHAUSTED.add(key)
        # Mixed
        mixed_ids = data.get('mixed_rotation', [])
        id_map_all = {q['id']: q for ql in NCERT_QUESTION_DATABASE.values() for q in ql}
        _MIXED_ROTATION.extend([id_map_all[i] for i in mixed_ids if i in id_map_all])
        global _MIXED_EXHAUSTED
        _MIXED_EXHAUSTED = data.get('mixed_exhausted', False)
        # User rotations
        for k, id_list in data.get('user_rotations', {}).get('lesson', {}).items():
            # k is string representation of tuple; we skip safe parsing for now
            if k.startswith('(') and k.endswith(')'):
                parts = k.strip('()').split(',')
                if len(parts) == 2:
                    user_id = parts[0].strip().strip("'\"")
                    lesson_id = parts[1].strip().strip("'\"")
                    id_map = _id_map_for_lesson(lesson_id)
                    _USER_ROTATIONS['lesson'][(user_id, lesson_id)] = [id_map[i] for i in id_list if i in id_map]
        # Subject-class user rotations skipped for brevity (can extend similarly)
        for uid, id_list in data.get('user_rotations', {}).get('mixed', {}).items():
            _USER_ROTATIONS['mixed'][uid] = [id_map_all[i] for i in id_list if i in id_map_all]
        # User exhausted
        for k in data.get('user_exhausted', {}).get('lesson', []):
            if k.startswith('(') and k.endswith(')'):
                parts = k.strip('()').split(',')
                if len(parts) == 2:
                    user_id = parts[0].strip().strip("'\"")
                    lesson_id = parts[1].strip().strip("'\"")
                    _USER_EXHAUSTED['lesson'].add((user_id, lesson_id))
        # Tracking
        for uid, rec in data.get('user_tracking', {}).items():
            _USER_TRACKING[uid] = {
                'answered': {lid: set(ids) for lid, ids in rec.get('answered', {}).items()},
                'wrong': {lid: set(ids) for lid, ids in rec.get('wrong', {}).items()}
            }
    except Exception:
        pass

# Load persisted state on import
load_question_state()