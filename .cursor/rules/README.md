# 📋 SKINMATE Cursor Rules Documentation

This directory contains comprehensive Cursor rules designed to guide development practices for the SKINMATE project. These rules implement advanced prompt engineering techniques and enforce critical coding standards.

## 🎯 Rule Categories

### 1. **coding-standards.mdc** (Always Applied)
- **Purpose**: Fundamental coding quality and security standards
- **Scope**: All Python files
- **Key Focus**: Security vulnerabilities, architectural patterns, error handling
- **Critical Issues**: Addresses hardcoded secrets, monolithic design, poor error handling

### 2. **flask-architecture.mdc** (Python Files)
- **Purpose**: Flask-specific architectural guidelines and best practices
- **Scope**: Python files (*.py)
- **Key Focus**: Blueprint organization, dependency injection, route design
- **Critical Issues**: Monolithic app.py refactoring, security implementations

### 3. **ai-ml-models.mdc** (ML Files)
- **Purpose**: AI/ML model management, security, and performance
- **Scope**: Python, pickle, and TensorFlow Lite files (*.py, *.pkl, *.tflite)
- **Key Focus**: Model loading, inference optimization, security
- **Critical Issues**: Global model variables, unsafe pickle loading

### 4. **database-operations.mdc** (Database Files)
- **Purpose**: Database security, performance, and best practices
- **Scope**: Python and SQL files (*.py, *.sql)
- **Key Focus**: ORM implementation, SQL injection prevention, query optimization
- **Critical Issues**: Raw SQL queries, lack of validation, performance bottlenecks

### 5. **security-compliance.mdc** (Always Applied)
- **Purpose**: Critical security compliance and vulnerability prevention
- **Scope**: All files
- **Key Focus**: Security vulnerabilities, authentication, secure coding
- **Critical Issues**: Hardcoded secrets, file upload vulnerabilities, authentication flaws

### 6. **project-structure.mdc** (Always Applied)
- **Purpose**: Project organization and structural guidelines
- **Scope**: All files
- **Key Focus**: Directory structure, module organization, configuration management
- **Critical Issues**: Monolithic structure, poor separation of concerns

## 🚨 Critical Issues Identified

### **IMMEDIATE ACTION REQUIRED**

1. **Security Vulnerabilities**
   - Hardcoded SECRET_KEY in app.py (line 38)
   - Unsafe file upload handling
   - No input validation or sanitization
   - Missing authentication protection

2. **Architectural Problems**
   - Monolithic app.py file (1061 lines)
   - No separation of concerns
   - Global variables for model management
   - Direct SQL queries without ORM

3. **Code Quality Issues**
   - Mixed responsibilities in route handlers
   - Poor error handling and logging
   - No comprehensive testing
   - Lack of type hints and documentation

## 📚 Advanced Prompt Engineering Techniques Applied

### 1. **Role-Based Instructions**
Each rule establishes clear expertise context and authority level for guidance.

### 2. **Constraint-Based Prompting**
Explicit "MUST DO" and "NEVER DO" sections with clear boundaries and requirements.

### 3. **Example-Driven Learning**
Comprehensive code examples showing both correct and incorrect implementations.

### 4. **Hierarchical Information Architecture**
Structured information flow from critical issues to implementation details.

### 5. **Context-Aware Guidance**
Rules reference specific files and line numbers from the actual codebase.

### 6. **Progressive Disclosure**
Information organized from high-level principles to detailed implementation.

## 🔧 Implementation Priority

### Phase 1: Security Fixes (CRITICAL)
1. Replace hardcoded secrets with environment variables
2. Implement secure file upload validation
3. Add input sanitization and authentication

### Phase 2: Architectural Refactoring (HIGH)
1. Extract blueprints from monolithic app.py
2. Implement service layer pattern
3. Create repository pattern for data access

### Phase 3: Quality Improvements (MEDIUM)
1. Add comprehensive test suite
2. Implement proper logging and monitoring
3. Create configuration management system

### Phase 4: Optimization (LOW)
1. Performance optimization
2. Advanced security features
3. Documentation and deployment automation

## 🎯 Using These Rules Effectively

### For Developers
- Rules are automatically applied based on file types and contexts
- Focus on the "MUST DO" and "NEVER DO" sections for immediate guidance
- Use code examples as templates for implementation

### For Code Reviews
- Reference specific rules when providing feedback
- Use rule violations as learning opportunities
- Ensure all critical security issues are addressed before approval

### For Project Managers
- Use rule compliance as quality metrics
- Prioritize fixes based on rule severity levels
- Track progress through rule-based checklists

## 🔍 Rule Maintenance

These rules should be updated as the project evolves:
- Add new patterns as they emerge
- Update examples based on actual implementation
- Refine constraints based on team feedback
- Maintain alignment with security best practices

---

Remember: These rules are not just guidelines—they represent critical requirements for building secure, maintainable, and scalable software. Compliance is mandatory, not optional.
